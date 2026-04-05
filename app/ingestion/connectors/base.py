import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field


@dataclass
class ConnectorResult:
    """Output produced by any ingestion connector."""

    text: str
    filename: str | None = None
    metadata: dict = field(default_factory=dict)


class BaseConnector(ABC):
    """Abstract base class for all ingestion connectors."""

    @abstractmethod
    async def extract(self, source: str) -> ConnectorResult:
        """Extract raw text and metadata from the given source.

        Args:
            source: A file path, URL, or identifier understood by this connector.

        Returns:
            ConnectorResult containing the extracted text and metadata.
        """
        ...


# ── External connector primitives ─────────────────────────────────────────────

@dataclass
class RawItem:
    """A single item fetched from an external system before transformation."""

    external_id: str
    item_type: str
    data: dict = field(default_factory=dict)


class ExternalConnector(ABC):
    """Abstract base class for external system connectors (Jira, ClickUp, etc.).

    Each subclass must declare a class-level ``connector_type`` string that
    matches the value stored in the ``connectors`` DB table.
    """

    connector_type: str

    @abstractmethod
    async def validate_credentials(self, credentials: dict) -> tuple[bool, str | None]:
        """Test whether the supplied credentials can reach the external API.

        Returns:
            (True, None) on success, (False, error_message) on failure.
        """
        ...

    @abstractmethod
    async def fetch_items(
        self,
        credentials: dict,
        config: dict,
        cursor: str | None,
    ) -> AsyncGenerator[RawItem, None]:
        """Yield raw items from the external system.

        Args:
            credentials: Decrypted credentials dict for this connector.
            config: Per-connector options (project keys, workspace IDs, etc.).
            cursor: Opaque string from the previous successful run's metadata,
                    used for incremental fetches. ``None`` on the first run.

        Yields:
            RawItem instances ready for transformation.
        """
        ...

    @abstractmethod
    def transform_item(
        self,
        item: RawItem,
        connector_id: uuid.UUID,
    ) -> ConnectorResult:
        """Convert a raw external item into a ConnectorResult for ingestion.

        The returned ``metadata`` dict must include at minimum:
            - ``connector_type``
            - ``connector_id``
            - ``external_id``
            - ``item_type``
            - ``source``  (formatted as ``"{connector_type}:{external_id}:{item_type}"``)
        """
        ...
