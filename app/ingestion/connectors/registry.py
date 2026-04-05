"""Connector registry — maps connector_type strings to ExternalConnector classes."""

from app.ingestion.connectors.base import ExternalConnector
from app.ingestion.connectors.clickup import ClickUpConnector
from app.ingestion.connectors.jira import JiraConnector
from app.ingestion.connectors.timedoctor import TimeDoctorConnector

CONNECTOR_REGISTRY: dict[str, type[ExternalConnector]] = {
    JiraConnector.connector_type: JiraConnector,
    ClickUpConnector.connector_type: ClickUpConnector,
    TimeDoctorConnector.connector_type: TimeDoctorConnector,
}


def get_connector(connector_type: str) -> ExternalConnector:
    """Instantiate and return the connector for the given type.

    Raises:
        ValueError: If ``connector_type`` is not registered.
    """
    cls = CONNECTOR_REGISTRY.get(connector_type)
    if cls is None:
        supported = ", ".join(sorted(CONNECTOR_REGISTRY))
        raise ValueError(
            f"Unknown connector type '{connector_type}'. "
            f"Supported: {supported}"
        )
    return cls()
