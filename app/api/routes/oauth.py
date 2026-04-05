"""
app/api/routes/oauth.py

One-click OAuth 2.0 connector authorization.

GET /oauth/{connector_type}/authorize  →  302 redirect to service login
GET /oauth/{connector_type}/callback   →  exchange code, create connector,
                                           redirect popup to frontend success page
"""
from __future__ import annotations

import json
import secrets
import urllib.parse

import httpx
import structlog
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import RedirectResponse
from redis.asyncio import Redis

from app.config import settings
from app.database.postgres import async_session_factory
from app.storage import connector_store
from app.utils.crypto import encrypt_credentials

logger = structlog.get_logger(__name__)
router = APIRouter()

_STATE_TTL = 600  # 10 minutes


# ── State helpers (CSRF protection) ──────────────────────────────────────────

async def _store_state(state: str, data: dict) -> None:
    r = Redis.from_url(settings.redis_url, decode_responses=True)
    await r.setex(f"oauth:state:{state}", _STATE_TTL, json.dumps(data))
    await r.aclose()


async def _pop_state(state: str) -> dict | None:
    r = Redis.from_url(settings.redis_url, decode_responses=True)
    raw = await r.getdel(f"oauth:state:{state}")
    await r.aclose()
    return json.loads(raw) if raw else None


def _redirect_uri(connector_type: str) -> str:
    return f"{settings.app_base_url}/oauth/{connector_type}/callback"


def _success_url(connector_id: str, connector_type: str) -> str:
    return (
        f"{settings.frontend_base_url}/oauth/success"
        f"?connector_id={connector_id}&type={connector_type}"
    )


def _error_url(message: str) -> str:
    return f"{settings.frontend_base_url}/oauth/error?message={urllib.parse.quote(message)}"


# ── Shared connector creation ─────────────────────────────────────────────────

async def _create_connector(
    connector_type: str,
    name: str,
    credentials: dict,
    config: dict | None = None,
) -> str:
    """Encrypt credentials and insert a new Connector row. Returns the connector ID."""
    credentials_enc = encrypt_credentials(credentials)
    async with async_session_factory() as session:
        connector = await connector_store.create_connector(
            session,
            name=name,
            connector_type=connector_type,
            credentials_enc=credentials_enc,
            config_json=json.dumps(config or {}),
            sync_schedule="0 */6 * * *",
            is_enabled=True,
        )
        await session.commit()
        return str(connector.id)


# ── Jira OAuth 2.0 (3LO) ─────────────────────────────────────────────────────

@router.get("/jira/authorize", include_in_schema=False)
async def jira_authorize() -> RedirectResponse:
    if not settings.jira_oauth_client_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Jira OAuth is not configured. Set JIRA_OAUTH_CLIENT_ID and JIRA_OAUTH_CLIENT_SECRET.",
        )
    state = secrets.token_urlsafe(32)
    await _store_state(state, {"connector_type": "jira"})
    url = (
        "https://auth.atlassian.com/authorize"
        "?audience=api.atlassian.com"
        f"&client_id={settings.jira_oauth_client_id}"
        "&scope=read%3Ajira-work%20read%3Ajira-user"
        f"&redirect_uri={urllib.parse.quote(_redirect_uri('jira'), safe='')}"
        f"&state={state}"
        "&response_type=code"
        "&prompt=consent"
    )
    return RedirectResponse(url, status_code=302)


@router.get("/jira/callback", include_in_schema=False)
async def jira_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    if error or not code or not state:
        return RedirectResponse(_error_url(error or "Jira authorization was denied."))

    state_data = await _pop_state(state)
    if not state_data:
        return RedirectResponse(_error_url("Invalid or expired state token. Please try again."))

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Exchange authorization code for tokens
            token_resp = await client.post(
                "https://auth.atlassian.com/oauth/token",
                json={
                    "grant_type": "authorization_code",
                    "client_id": settings.jira_oauth_client_id,
                    "client_secret": settings.jira_oauth_client_secret,
                    "code": code,
                    "redirect_uri": _redirect_uri("jira"),
                },
            )
            token_resp.raise_for_status()
            token_data = token_resp.json()

            # Discover cloud instances accessible to this token
            resources_resp = await client.get(
                "https://api.atlassian.com/oauth/token/accessible-resources",
                headers={"Authorization": f"Bearer {token_data['access_token']}"},
            )
            resources_resp.raise_for_status()
            resources = resources_resp.json()

        if not resources:
            return RedirectResponse(_error_url("No Jira cloud instance found on this account."))

        resource = resources[0]
        cloud_id: str = resource["id"]
        site_name: str = resource.get("name", "Jira")

        credentials = {
            "auth_type": "oauth",
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token", ""),
            "cloud_id": cloud_id,
            "base_url": f"https://api.atlassian.com/ex/jira/{cloud_id}",
        }
        connector_id = await _create_connector("jira", f"Jira — {site_name}", credentials)
        logger.info("jira_oauth_connector_created", connector_id=connector_id, site=site_name)
        return RedirectResponse(_success_url(connector_id, "jira"))

    except httpx.HTTPStatusError as exc:
        msg = f"Jira OAuth failed: HTTP {exc.response.status_code}"
        logger.error("jira_oauth_error", error=str(exc))
        return RedirectResponse(_error_url(msg))
    except Exception as exc:
        logger.error("jira_oauth_error", error=str(exc))
        return RedirectResponse(_error_url(f"Jira OAuth failed: {str(exc)[:120]}"))


# ── ClickUp OAuth 2.0 ────────────────────────────────────────────────────────

@router.get("/clickup/authorize", include_in_schema=False)
async def clickup_authorize() -> RedirectResponse:
    if not settings.clickup_oauth_client_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="ClickUp OAuth is not configured. Set CLICKUP_OAUTH_CLIENT_ID and CLICKUP_OAUTH_CLIENT_SECRET.",
        )
    state = secrets.token_urlsafe(32)
    await _store_state(state, {"connector_type": "clickup"})
    url = (
        "https://app.clickup.com/api"
        f"?client_id={settings.clickup_oauth_client_id}"
        f"&redirect_uri={urllib.parse.quote(_redirect_uri('clickup'), safe='')}"
    )
    return RedirectResponse(url, status_code=302)


@router.get("/clickup/callback", include_in_schema=False)
async def clickup_callback(
    code: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    if error or not code:
        return RedirectResponse(_error_url(error or "ClickUp authorization was denied."))

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            token_resp = await client.post(
                "https://api.clickup.com/api/v2/oauth/token",
                params={
                    "client_id": settings.clickup_oauth_client_id,
                    "client_secret": settings.clickup_oauth_client_secret,
                    "code": code,
                },
            )
            token_resp.raise_for_status()
            token_data = token_resp.json()

            # Get user info to name the connector
            user_resp = await client.get(
                "https://api.clickup.com/api/v2/user",
                headers={"Authorization": token_data["access_token"]},
            )
            user_resp.raise_for_status()
            user_data = user_resp.json()

        username: str = user_data.get("user", {}).get("username", "")
        credentials = {
            "api_token": token_data["access_token"],
            "auth_type": "oauth",
        }
        name = f"ClickUp — {username}" if username else "ClickUp"
        connector_id = await _create_connector("clickup", name, credentials)
        logger.info("clickup_oauth_connector_created", connector_id=connector_id, user=username)
        return RedirectResponse(_success_url(connector_id, "clickup"))

    except httpx.HTTPStatusError as exc:
        msg = f"ClickUp OAuth failed: HTTP {exc.response.status_code}"
        logger.error("clickup_oauth_error", error=str(exc))
        return RedirectResponse(_error_url(msg))
    except Exception as exc:
        logger.error("clickup_oauth_error", error=str(exc))
        return RedirectResponse(_error_url(f"ClickUp OAuth failed: {str(exc)[:120]}"))


# ── Time Doctor OAuth 2.0 ─────────────────────────────────────────────────────

@router.get("/timedoctor/authorize", include_in_schema=False)
async def timedoctor_authorize() -> RedirectResponse:
    if not settings.timedoctor_oauth_client_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Time Doctor OAuth is not configured. Set TIMEDOCTOR_OAUTH_CLIENT_ID and TIMEDOCTOR_OAUTH_CLIENT_SECRET.",
        )
    state = secrets.token_urlsafe(32)
    await _store_state(state, {"connector_type": "timedoctor"})
    url = (
        "https://api2.timedoctor.com/oauth/v2/auth"
        "?response_type=code"
        f"&client_id={settings.timedoctor_oauth_client_id}"
        f"&redirect_uri={urllib.parse.quote(_redirect_uri('timedoctor'), safe='')}"
        "&scope=openid"
        f"&state={state}"
    )
    return RedirectResponse(url, status_code=302)


@router.get("/timedoctor/callback", include_in_schema=False)
async def timedoctor_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    if error or not code or not state:
        return RedirectResponse(_error_url(error or "Time Doctor authorization was denied."))

    state_data = await _pop_state(state)
    if not state_data:
        return RedirectResponse(_error_url("Invalid or expired state token. Please try again."))

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            token_resp = await client.post(
                "https://api2.timedoctor.com/oauth/v2/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": settings.timedoctor_oauth_client_id,
                    "client_secret": settings.timedoctor_oauth_client_secret,
                    "redirect_uri": _redirect_uri("timedoctor"),
                    "code": code,
                },
            )
            token_resp.raise_for_status()
            token_data = token_resp.json()

            # Discover the user's company
            companies_resp = await client.get(
                "https://webapi.timedoctor.com/api/1.0/companies",
                headers={"Authorization": f"Bearer {token_data['access_token']}"},
            )
            companies_resp.raise_for_status()
            accounts = companies_resp.json().get("accounts", [])

        company_id = str(accounts[0]["company_id"]) if accounts else ""
        company_name: str = accounts[0].get("name", "") if accounts else ""

        credentials = {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token", ""),
            "company_id": company_id,
            "auth_type": "oauth",
        }
        name = f"Time Doctor — {company_name}" if company_name else "Time Doctor"
        connector_id = await _create_connector("timedoctor", name, credentials)
        logger.info("timedoctor_oauth_connector_created", connector_id=connector_id, company=company_name)
        return RedirectResponse(_success_url(connector_id, "timedoctor"))

    except httpx.HTTPStatusError as exc:
        msg = f"Time Doctor OAuth failed: HTTP {exc.response.status_code}"
        logger.error("timedoctor_oauth_error", error=str(exc))
        return RedirectResponse(_error_url(msg))
    except Exception as exc:
        logger.error("timedoctor_oauth_error", error=str(exc))
        return RedirectResponse(_error_url(f"Time Doctor OAuth failed: {str(exc)[:120]}"))
