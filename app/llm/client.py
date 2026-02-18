from typing import Any

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class OllamaClient:
    """Async client for the Ollama REST API."""

    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        timeout: float = settings.ollama_request_timeout,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout, connect=10.0)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )

    # ── Generate (single-turn completion) ────────────────
    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Send a single prompt and return the generated text."""
        model = model or settings.ollama_synthesis_model
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with self._client() as client:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        logger.debug("ollama_generate", model=model, prompt_len=len(prompt))
        return data["response"]

    # ── Chat (multi-turn) ────────────────────────────────
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat conversation and return the assistant reply."""
        model = model or settings.ollama_synthesis_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with self._client() as client:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        logger.debug("ollama_chat", model=model, turns=len(messages))
        return data["message"]["content"]

    # ── Embeddings ───────────────────────────────────────
    async def embed(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """Generate an embedding vector for the given text."""
        model = model or settings.embedding_model
        payload: dict[str, Any] = {
            "model": model,
            "input": text,
        }
        async with self._client() as client:
            response = await client.post("/api/embed", json=payload)
            response.raise_for_status()
            data = response.json()
        logger.debug("ollama_embed", model=model, text_len=len(text))
        return data["embeddings"][0]

    # ── Batch Embeddings ─────────────────────────────────
    async def embed_batch(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts."""
        model = model or settings.embedding_model
        payload: dict[str, Any] = {
            "model": model,
            "input": texts,
        }
        async with self._client() as client:
            response = await client.post("/api/embed", json=payload)
            response.raise_for_status()
            data = response.json()
        logger.debug("ollama_embed_batch", model=model, count=len(texts))
        return data["embeddings"]

    # ── Code Generation (uses codellama) ─────────────────
    async def generate_code(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """Generate code (Cypher/SQL) using the codegen model."""
        return await self.generate(
            prompt=prompt,
            model=settings.ollama_codegen_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── Health Check ─────────────────────────────────────
    async def is_healthy(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            async with self._client() as client:
                response = await client.get("/")
                return response.status_code == 200
        except httpx.HTTPError:
            return False


# Module-level singleton
ollama_client = OllamaClient()
