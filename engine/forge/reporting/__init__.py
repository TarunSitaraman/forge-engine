"""Rendering diagnostics for humans who are not standing at a terminal."""

from .html import render_html
from .markdown import render_markdown

__all__ = ["render_html", "render_markdown"]
