"""Time Doctor connector — fetches work logs via the Time Doctor REST API v1."""

import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone

import httpx
import structlog

from app.ingestion.connectors.base import ConnectorResult, ExternalConnector, RawItem

logger = structlog.get_logger(__name__)

_TD_API = "https://webapi.timedoctor.com"
_AUTH_URL = f"{_TD_API}/oauth/v2/token"
_DEFAULT_DAYS_BACK = 7


class TimeDoctorConnector(ExternalConnector):
    """Syncs Time Doctor work logs into TemporalDB."""

    connector_type = "timedoctor"

    # ── Auth ──────────────────────────────────────────────────────────────────

    async def _get_access_token(self, credentials: dict) -> str:
        """Return a valid access token, refreshing if needed."""
        # If a non-expired access token is already in credentials, reuse it
        if credentials.get("access_token"):
            return credentials["access_token"]

        if credentials.get("refresh_token"):
            return await self._refresh_token(credentials)

        return await self._password_grant(credentials)

    async def _password_grant(self, credentials: dict) -> str:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                _AUTH_URL,
                data={
                    "grant_type": "password",
                    "client_id": "webapi",
                    "client_secret": "secret",
                    "username": credentials["email"],
                    "password": credentials["password"],
                },
            )
            resp.raise_for_status()
            data = resp.json()
        token: str = data["access_token"]
        # Callers may persist these back to credentials_enc via the sync service
        credentials["access_token"] = token
        credentials["refresh_token"] = data.get("refresh_token", "")
        return token

    async def _refresh_token(self, credentials: dict) -> str:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                _AUTH_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": "webapi",
                    "client_secret": "secret",
                    "refresh_token": credentials["refresh_token"],
                },
            )
            if resp.status_code == 400:
                # Refresh token expired — fall back to password grant
                logger.info("timedoctor_refresh_token_expired_retrying_password_grant")
                credentials.pop("refresh_token", None)
                credentials.pop("access_token", None)
                return await self._password_grant(credentials)
            resp.raise_for_status()
            data = resp.json()
        token: str = data["access_token"]
        credentials["access_token"] = token
        credentials["refresh_token"] = data.get("refresh_token", credentials.get("refresh_token", ""))
        return token

    def _authed_client(self, access_token: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=_TD_API,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30.0,
        )

    # ── Validate ──────────────────────────────────────────────────────────────

    async def validate_credentials(self, credentials: dict) -> tuple[bool, str | None]:
        try:
            token = await self._get_access_token(credentials)
            company_id = credentials["company_id"]
            async with self._authed_client(token) as client:
                resp = await client.get(f"/api/1.0/{company_id}/users")
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
        company_id: str = credentials["company_id"]
        user_ids: list[str] = config.get("user_ids", [])
        project_ids: list[str] = config.get("project_ids", [])
        days_back: int = config.get("days_back", _DEFAULT_DAYS_BACK)

        # Determine date range
        end_date = datetime.now(timezone.utc).date()
        if cursor:
            # cursor is an ISO date string (YYYY-MM-DD)
            from datetime import date
            start_date = date.fromisoformat(cursor)
        else:
            start_date = end_date - timedelta(days=days_back)

        token = await self._get_access_token(credentials)

        async with self._authed_client(token) as client:
            # Build project name map for richer text output
            project_name_map = await self._get_project_names(
                client, company_id, project_ids
            )
            user_name_map = await self._get_user_names(client, company_id, user_ids)

            # Fetch worklogs
            params: dict = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
            if user_ids:
                params["user_ids"] = ",".join(user_ids)
            if project_ids:
                params["project_ids"] = ",".join(project_ids)

            resp = await client.get(f"/api/1.0/{company_id}/worklogs", params=params)
            resp.raise_for_status()
            worklogs_data = resp.json()

            for worklog in worklogs_data.get("worklogs", {}).get("items", []):
                # Enrich with resolved names
                worklog["_project_name"] = project_name_map.get(
                    str(worklog.get("project_id", "")), "Unknown Project"
                )
                worklog["_user_name"] = user_name_map.get(
                    str(worklog.get("user_id", "")), "Unknown User"
                )
                yield RawItem(
                    external_id=str(worklog.get("id", "")),
                    item_type="worklog",
                    data=worklog,
                )

    async def _get_project_names(
        self,
        client: httpx.AsyncClient,
        company_id: str,
        project_ids: list[str],
    ) -> dict[str, str]:
        resp = await client.get(f"/api/1.0/{company_id}/projects")
        if resp.status_code != 200:
            return {}
        projects = resp.json().get("projects", {}).get("items", [])
        name_map = {str(p["id"]): p.get("name", "") for p in projects}
        if project_ids:
            return {k: v for k, v in name_map.items() if k in project_ids}
        return name_map

    async def _get_user_names(
        self,
        client: httpx.AsyncClient,
        company_id: str,
        user_ids: list[str],
    ) -> dict[str, str]:
        resp = await client.get(f"/api/1.0/{company_id}/users")
        if resp.status_code != 200:
            return {}
        users = resp.json().get("users", {}).get("items", [])
        name_map = {str(u["id"]): u.get("name", u.get("email", "")) for u in users}
        if user_ids:
            return {k: v for k, v in name_map.items() if k in user_ids}
        return name_map

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform_item(self, item: RawItem, connector_id: uuid.UUID) -> ConnectorResult:
        if item.item_type == "worklog":
            return self._transform_worklog(item, connector_id)
        else:
            raise ValueError(f"Unknown Time Doctor item type: {item.item_type}")

    def _transform_worklog(
        self, item: RawItem, connector_id: uuid.UUID
    ) -> ConnectorResult:
        d = item.data
        worklog_id = str(d.get("id", item.external_id))
        user_name = d.get("_user_name", "Unknown User")
        task_name = d.get("task_name") or d.get("task", {}).get("name", "Unknown Task")
        project_name = d.get("_project_name", "Unknown Project")

        start_time = d.get("start_time", "")
        end_time = d.get("end_time", "")
        length: int = d.get("length", 0)  # seconds
        duration_str = _format_duration(length)
        date = d.get("date", "")
        activity_pct = d.get("activity_percent", d.get("active", ""))

        text = (
            f"Time log: {user_name} worked on '{task_name}' "
            f"(project: {project_name}) "
            f"from {start_time} to {end_time} ({duration_str}).\n"
            f"Activity: {activity_pct}%\n"
            f"Date: {date}"
        ).strip()

        source = f"timedoctor:{worklog_id}:worklog"
        return ConnectorResult(
            text=text,
            filename=f"worklog-{worklog_id}.timedoctor",
            metadata={
                "connector_type": "timedoctor",
                "connector_id": str(connector_id),
                "external_id": worklog_id,
                "item_type": "worklog",
                "source": source,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "user": user_name,
                "project": project_name,
                "task": task_name,
                "duration_seconds": length,
                "date": date,
            },
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_duration(seconds: int) -> str:
    if seconds <= 0:
        return "0s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return " ".join(parts)
