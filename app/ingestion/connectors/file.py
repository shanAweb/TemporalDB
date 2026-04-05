import struct
from pathlib import Path

import fitz  # PyMuPDF
import structlog
from docx import Document

from app.ingestion.connectors.base import BaseConnector, ConnectorResult

logger = structlog.get_logger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


class FileConnector(BaseConnector):
    """Extracts text from local files: PDF, DOCX, TXT, and Markdown."""

    async def extract(self, source: str) -> ConnectorResult:
        """Extract text from a local file.

        Args:
            source: Absolute or relative path to the file.

        Returns:
            ConnectorResult with extracted text and file metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        ext = path.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        logger.info("file_connector_extracting", path=str(path), extension=ext)

        if ext == ".pdf":
            return self._extract_pdf(path)
        elif ext == ".docx":
            return self._extract_docx(path)
        else:
            return self._extract_text(path)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_pdf(self, path: Path) -> ConnectorResult:
        """Extract text from a PDF using PyMuPDF."""
        doc = fitz.open(str(path))
        pages_text: list[str] = []

        for page in doc:
            pages_text.append(page.get_text())

        text = "\n".join(pages_text)
        metadata = {
            "page_count": doc.page_count,
            "file_size_bytes": path.stat().st_size,
            "pdf_metadata": doc.metadata,
        }
        doc.close()

        logger.debug("pdf_extracted", filename=path.name, pages=len(pages_text))
        return ConnectorResult(text=text, filename=path.name, metadata=metadata)

    def _extract_docx(self, path: Path) -> ConnectorResult:
        """Extract text from a DOCX file using python-docx."""
        doc = Document(str(path))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        text = "\n".join(paragraphs)
        metadata = {
            "paragraph_count": len(paragraphs),
            "file_size_bytes": path.stat().st_size,
        }

        logger.debug("docx_extracted", filename=path.name, paragraphs=len(paragraphs))
        return ConnectorResult(text=text, filename=path.name, metadata=metadata)

    def _extract_text(self, path: Path) -> ConnectorResult:
        """Extract text from a plain-text or Markdown file."""
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata = {
            "file_size_bytes": path.stat().st_size,
            "encoding": "utf-8",
        }

        logger.debug("text_extracted", filename=path.name, chars=len(text))
        return ConnectorResult(text=text, filename=path.name, metadata=metadata)
