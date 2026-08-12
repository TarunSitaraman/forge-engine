"""Source adapters — acquisition only.

Adapters turn external material into located text. They do not extract
concepts, discover relationships, or call an LLM.
"""

from .base import AcquisitionResult, SourceAdapter, TextBlock, failed, normalize_text
from .markdown_adapter import MarkdownAdapter
from .pdf_adapter import PdfAdapter
from .registry import AdapterRegistry

__all__ = [
    "SourceAdapter",
    "AcquisitionResult",
    "TextBlock",
    "MarkdownAdapter",
    "PdfAdapter",
    "AdapterRegistry",
    "failed",
    "normalize_text",
]
