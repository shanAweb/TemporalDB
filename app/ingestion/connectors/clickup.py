"""ClickUp connector — fetches tasks and comments via the ClickUp REST API v2."""

import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import httpx
import structlog

from app.ingestion.connectors.base import ConnectorResult, ExternalConnector, RawItem

logger = structlog.get_logger(__name__)

_CLICKUP_API = "https://api.clickup.com/api/v2"
_DEFAULT_PAGE_SIZE = 100


class ClickUpConnector(ExternalConnector):
    """Syncs ClickUp tasks and comments into TemporalDB."""

    connector_type = "clickup"

    # ── Credentials & config helpers ──────────────────────────────────────────

    @staticmethod
    def _client(credentials: dict) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=_CLICKUP_API,
            headers={
                "Authorization": credentials["api_token"],
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    # ── Validate ──────────────────────────────────────────────────────────────

    async def validate_credentials(self, credentials: dict) -> tuple[bool, str | None]:
        try:
            async with self._client(credentials) as client:
                resp = await client.get("/user")
                resp.raise_for_status()
            return True, None
        except httpx.HTTPStatusError as exc:
            return False, f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        except Exception as exc:
            return False, str(exc)

    # ── Fetch ─────────────────────────────────────────────────────────────────

    async def fetch_items(
        self,
        credentials: dict,
        config: dict,
        cursor: str | None,
    ) -> AsyncGenerator[RawItem, None]:
        team_id: str = config["team_id"]
        space_ids: list[str] = config.get("space_ids", [])
        list_ids: list[str] = config.get("list_ids", [])
        include_comments: bool = config.get("include_comments", True)
        include_archived: bool = config.get("include_archived", False)
        filter_statuses: list[str] = config.get("statuses", [])
        days_back: int = config.get("updated_since_days", 30)

        # cursor is a Unix ms timestamp string
        date_updated_gt: int | None = int(cursor) if cursor else None
        if date_updated_gt is None:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
            date_updated_gt = int(cutoff.timestamp() * 1000)

        async with self._client(credentials) as client:
            # Determine which lists to iterate
            target_list_ids: list[str] = list(list_ids)

            if not target_list_ids:
                # Resolve from spaces or fetch all spaces for the team
                target_space_ids = list(space_ids)
                if not target_space_ids:
                    target_space_ids = await self._get_space_ids(client, team_id)

                for space_id in target_space_ids:
                    async for list_id in self._get_list_ids(
                        client, space_id, include_archived
                    ):
                        target_list_ids.append(list_id)

            # Fetch tasks for each list
            for list_id in target_list_ids:
                async for task_item in self._fetch_tasks(
                    client,
                    list_id,
                    date_updated_gt,
                    include_archived,
                    filter_statuses,
                ):
                    yield task_item

                    if include_comments:
                        async for comment_item in self._fetch_comments(
                            client, task_item.external_id
                        ):
                            yield comment_item

    async def _get_space_ids(
        self, client: httpx.AsyncClient, team_id: str
    ) -> list[str]:
        resp = await client.get(f"/team/{team_id}/space", params={"archived": False})
        if resp.status_code != 200:
            logger.warning("clickup_spaces_fetch_failed", team=team_id, status=resp.status_code)
            return []
        return [s["id"] for s in resp.json().get("spaces", [])]

    async def _get_list_ids(
        self,
        client: httpx.AsyncClient,
        space_id: str,
        include_archived: bool,
    ) -> AsyncGenerator[str, None]:
        resp = await client.get(
            f"/space/{space_id}/list", params={"archived": include_archived}
        )
        if resp.status_code != 200:
            return
        for lst in resp.json().get("lists", []):
            yield lst["id"]

    async def _fetch_tasks(
        self,
        client: httpx.AsyncClient,
        list_id: str,
        date_updated_gt: int,
        include_archived: bool,
        filter_statuses: list[str],
    ) -> AsyncGenerator[RawItem, None]:
        page = 0
        while True:
            params: dict = {
                "page": page,
                "include_closed": True,
                "subtasks": True,
                "archived": include_archived,
                "date_updated_gt": date_updated_gt,
                "order_by": "updated",
            }
            if filter_statuses:
                params["statuses[]"] = filter_statuses

            resp = await client.get(f"/list/{list_id}/task", params=params)
            if resp.status_code != 200:
                logger.warning(
                    "clickup_tasks_fetch_failed",
                    list_id=list_id,
                    status=resp.status_code,
                )
                break

            tasks: list[dict] = resp.json().get("tasks", [])
            if not tasks:
                break

            for task in tasks:
                yield RawItem(
                    external_id=task["id"],
                    item_type="task",
                    data=task,
                )

            page += 1
            if resp.json().get("last_page", True):
                break

    async def _fetch_comments(
        self, client: httpx.AsyncClient, task_id: str
    ) -> AsyncGenerator[RawItem, None]:
        resp = await client.get(f"/task/{task_id}/comment")
        if resp.status_code != 200:
            return
        for comment in resp.json().get("comments", []):
            yield RawItem(
                external_id=str(comment["id"]),
                item_type="comment",
                data={**comment, "_task_id": task_id},
            )

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform_item(self, item: RawItem, connector_id: uuid.UUID) -> ConnectorResult:
        if item.item_type == "task":
            return self._transform_task(item, connector_id)
        elif item.item_type == "comment":
            return self._transform_comment(item, connector_id)
        else:
            raise ValueError(f"Unknown ClickUp item type: {item.item_type}")

    def _transform_task(self, item: RawItem, connector_id: uuid.UUID) -> ConnectorResult:
        d = item.data
        task_id = d.get("id", item.external_id)
        name = d.get("name", "")
        description = d.get("description") or ""
        status = (d.get("status") or {}).get("status", "Unknown")
        priority = (d.get("priority") or {}).get("priority", "None")
        assignees = ", ".join(a.get("username", "") for a in d.get("assignees", [])) or "Unassigned"
        tags = ", ".join(t.get("name", "") for t in d.get("tags", [])) or "None"
        list_name = (d.get("list") or {}).get("name", "")
        space_name = (d.get("space") or {}).get("name", "")
        due_date = _ms_to_iso(d.get("due_date"))
        date_created = _ms_to_iso(d.get("date_created"))
        date_updated = _ms_to_iso(d.get("date_updated"))

        text = (
            f"[{task_id}] {name}\n\n"
            f"{description}\n\n"
            f"Status: {status}\n"
            f"Priority: {priority}\n"
            f"Assignees: {assignees}\n"
            f"Tags: {tags}\n"
            f"List: {list_name}\n"
            f"Space: {space_name}\n"
            f"Due: {due_date}\n"
            f"Created: {date_created}\n"
            f"Updated: {date_updated}"
        ).strip()

        source = f"clickup:{task_id}:task"
        return ConnectorResult(
            text=text,
            filename=f"{task_id}.clickup",
            metadata={
                "connector_type": "clickup",
                "connector_id": str(connector_id),
                "external_id": task_id,
                "item_type": "task",
                "source": source,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "list": list_name,
                "space": space_name,
            },
        )

    def _transform_comment(self, item: RawItem, connector_id: uuid.UUID) -> ConnectorResult:
        d = item.data
        comment_id = str(d.get("id", item.external_id))
        task_id = d.get("_task_id", "UNKNOWN")
        user = (d.get("user") or {}).get("username", "Unknown")
        comment_text = _extract_comment_text(d.get("comment_text") or d.get("comment", ""))
        date = _ms_to_iso(d.get("date"))

        text = (
            f"Comment on task [{task_id}] by {user} ({date}):\n\n{comment_text}"
        ).strip()

        source = f"clickup:{comment_id}:comment"
        return ConnectorResult(
            text=text,
            filename=f"{task_id}-comment-{comment_id}.clickup",
            metadata={
                "connector_type": "clickup",
                "connector_id": str(connector_id),
                "external_id": comment_id,
                "item_type": "comment",
                "source": source,
                "task_id": task_id,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ms_to_iso(ms_str: str | int | None) -> str:
    """Convert a Unix millisecond timestamp to an ISO 8601 string."""
    if ms_str is None:
        return ""
    try:
        ts = int(ms_str) / 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (ValueError, TypeError):
        return str(ms_str)


def _extract_comment_text(value: str | list | None) -> str:
    """ClickUp comment_text can be a plain string or a list of content blocks."""
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(p for p in parts if p)
    return str(value)
