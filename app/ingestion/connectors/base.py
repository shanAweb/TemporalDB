from abc import ABC, abstractmethod
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
