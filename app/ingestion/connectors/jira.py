"""Jira connector — fetches issues, comments, and sprints via the Jira REST API v3."""

import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import httpx
import structlog

from app.ingestion.connectors.base import ConnectorResult, ExternalConnector, RawItem

logger = structlog.get_logger(__name__)

_JIRA_API = "/rest/api/3"
_DEFAULT_PAGE_SIZE = 100


class JiraConnector(ExternalConnector):
    """Syncs Jira issues, comments, and sprints into TemporalDB."""

    connector_type = "jira"

    # ── Credentials & config helpers ──────────────────────────────────────────

    @staticmethod
    def _client(credentials: dict) -> httpx.AsyncClient:
        base_url = credentials["base_url"].rstrip("/")
        if credentials.get("auth_type") == "oauth":
            return httpx.AsyncClient(
                base_url=base_url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {credentials['access_token']}",
                },
                timeout=30.0,
            )
        auth = httpx.BasicAuth(credentials["email"], credentials["api_token"])
        return httpx.AsyncClient(
            base_url=base_url,
            auth=auth,
            headers={"Accept": "application/json"},
            timeout=30.0,
        )

    # ── Validate ──────────────────────────────────────────────────────────────

    async def validate_credentials(self, credentials: dict) -> tuple[bool, str | None]:
        try:
            async with self._client(credentials) as client:
                resp = await client.get(f"{_JIRA_API}/myself")
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
        project_keys: list[str] = config.get("project_keys", [])
        issue_types: list[str] = config.get(
            "issue_types", ["Story", "Bug", "Task", "Epic", "Sub-task"]
        )
        include_comments: bool = config.get("include_comments", True)
        include_sprints: bool = config.get("include_sprints", True)
        page_size: int = config.get("max_results_per_page", _DEFAULT_PAGE_SIZE)
        days_back: int = config.get("updated_since_days", 30)

        # Build JQL
        parts: list[str] = []
        if project_keys:
            keys_str = ", ".join(f'"{k}"' for k in project_keys)
            parts.append(f"project in ({keys_str})")
        if issue_types:
            types_str = ", ".join(f'"{t}"' for t in issue_types)
            parts.append(f"issuetype in ({types_str})")
        if cursor:
            # cursor is an ISO timestamp from the previous run
            parts.append(f'updated >= "{cursor}"')
        else:
            parts.append(f"updated >= -{days_back}d")
        parts.append("ORDER BY updated ASC")
        jql = " AND ".join(parts[:-1]) + " " + parts[-1]

        async with self._client(credentials) as client:
            # ── Issues ────────────────────────────────────────────────────────
            start_at = 0
            while True:
                resp = await client.get(
                    f"{_JIRA_API}/search",
                    params={
                        "jql": jql,
                        "startAt": start_at,
                        "maxResults": page_size,
                        "fields": (
                            "summary,description,status,assignee,priority,"
                            "labels,issuetype,project,created,updated,comment"
                        ),
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                issues: list[dict] = data.get("issues", [])

                if not issues:
                    break

                for issue in issues:
                    yield RawItem(
                        external_id=issue["key"],
                        item_type="issue",
                        data=issue,
                    )

                    if include_comments:
                        async for comment_item in self._fetch_comments(
                            client, issue["key"]
                        ):
                            yield comment_item

                start_at += len(issues)
                if start_at >= data.get("total", 0):
                    break

            # ── Sprints ───────────────────────────────────────────────────────
            if include_sprints:
                async for sprint_item in self._fetch_sprints(
                    client, project_keys, page_size
                ):
                    yield sprint_item

    async def _fetch_comments(
        self, client: httpx.AsyncClient, issue_key: str
    ) -> AsyncGenerator[RawItem, None]:
        start_at = 0
        while True:
            resp = await client.get(
                f"{_JIRA_API}/issue/{issue_key}/comment",
                params={"startAt": start_at, "maxResults": 50},
            )
            if resp.status_code != 200:
                logger.warning("jira_comment_fetch_failed", issue=issue_key, status=resp.status_code)
                return
            data = resp.json()
            comments: list[dict] = data.get("comments", [])
            for comment in comments:
                yield RawItem(
                    external_id=comment["id"],
                    item_type="comment",
                    data={**comment, "_issue_key": issue_key},
                )
            start_at += len(comments)
            if start_at >= data.get("total", 0):
                break

    async def _fetch_sprints(
        self,
        client: httpx.AsyncClient,
        project_keys: list[str],
        page_size: int,
    ) -> AsyncGenerator[RawItem, None]:
        # Fetch boards then sprints for each board
        resp = await client.get(
            "/rest/agile/1.0/board",
            params={"maxResults": page_size},
        )
        if resp.status_code != 200:
            return
        boards = resp.json().get("values", [])
        for board in boards:
            board_project = board.get("location", {}).get("projectKey", "")
            if project_keys and board_project not in project_keys:
                continue
            sprint_resp = await client.get(
                f"/rest/agile/1.0/board/{board['id']}/sprint",
                params={"maxResults": 50},
            )
            if sprint_resp.status_code != 200:
                continue
            for sprint in sprint_resp.json().get("values", []):
                yield RawItem(
                    external_id=str(sprint["id"]),
                    item_type="sprint",
                    data=sprint,
                )

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform_item(self, item: RawItem, connector_id: uuid.UUID) -> ConnectorResult:
        if item.item_type == "issue":
            return self._transform_issue(item, connector_id)
        elif item.item_type == "comment":
            return self._transform_comment(item, connector_id)
        elif item.item_type == "sprint":
            return self._transform_sprint(item, connector_id)
        else:
            raise ValueError(f"Unknown Jira item type: {item.item_type}")

    def _transform_issue(self, item: RawItem, connector_id: uuid.UUID) -> ConnectorResult:
        d = item.data
        fields = d.get("fields", {})
        key = d.get("key", item.external_id)
        summary = fields.get("summary", "")
        description = _extract_text_from_adf(fields.get("description")) or ""
        status = fields.get("status", {}).get("name", "Unknown")
        assignee = (fields.get("assignee") or {}).get("displayName", "Unassigned")
        priority = (fields.get("priority") or {}).get("name", "None")
        issue_type = (fields.get("issuetype") or {}).get("name", "")
        labels = ", ".join(fields.get("labels", [])) or "None"
        project = (fields.get("project") or {}).get("name", "")
        created = fields.get("created", "")
        updated = fields.get("updated", "")

        text = (
            f"[{key}] {summary}\n\n"
            f"{description}\n\n"
            f"Type: {issue_type}\n"
            f"Status: {status}\n"
            f"Assignee: {assignee}\n"
            f"Priority: {priority}\n"
            f"Labels: {labels}\n"
            f"Project: {project}\n"
            f"Created: {created}\n"
            f"Updated: {updated}"
        ).strip()

        source = f"jira:{key}:issue"
        return ConnectorResult(
            text=text,
            filename=f"{key}.jira",
            metadata={
                "connector_type": "jira",
                "connector_id": str(connector_id),
                "external_id": key,
                "item_type": "issue",
                "source": source,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "project": project,
                "status": status,
            },
        )

    def _transform_comment(self, item: RawItem, connector_id: uuid.UUID) -> ConnectorResult:
        d = item.data
        issue_key = d.get("_issue_key", "UNKNOWN")
        comment_id = d.get("id", item.external_id)
        author = (d.get("author") or {}).get("displayName", "Unknown")
        body = _extract_text_from_adf(d.get("body")) or ""
        created = d.get("created", "")

        text = (
            f"Comment on [{issue_key}] by {author} ({created}):\n\n{body}"
        ).strip()

        source = f"jira:{comment_id}:comment"
        return ConnectorResult(
            text=text,
            filename=f"{issue_key}-comment-{comment_id}.jira",
            metadata={
                "connector_type": "jira",
                "connector_id": str(connector_id),
                "external_id": comment_id,
                "item_type": "comment",
                "source": source,
                "issue_key": issue_key,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _transform_sprint(self, item: RawItem, connector_id: uuid.UUID) -> ConnectorResult:
        d = item.data
        sprint_id = str(d.get("id", item.external_id))
        name = d.get("name", "")
        goal = d.get("goal", "") or ""
        state = d.get("state", "")
        start_date = d.get("startDate", "")
        end_date = d.get("endDate", "")

        text = (
            f"Sprint: {name}\n"
            f"Goal: {goal}\n"
            f"State: {state}\n"
            f"Start: {start_date}\n"
            f"End: {end_date}"
        ).strip()

        source = f"jira:{sprint_id}:sprint"
        return ConnectorResult(
            text=text,
            filename=f"sprint-{sprint_id}.jira",
            metadata={
                "connector_type": "jira",
                "connector_id": str(connector_id),
                "external_id": sprint_id,
                "item_type": "sprint",
                "source": source,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text_from_adf(node: dict | str | None) -> str:
    """Recursively extract plain text from Jira's Atlassian Document Format (ADF)."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""

    node_type = node.get("type", "")
    text_parts: list[str] = []

    if node_type == "text":
        return node.get("text", "")

    for child in node.get("content", []):
        part = _extract_text_from_adf(child)
        if part:
            text_parts.append(part)

    separator = "\n" if node_type in ("paragraph", "heading", "bulletList", "orderedList", "listItem") else " "
    return separator.join(text_parts)
