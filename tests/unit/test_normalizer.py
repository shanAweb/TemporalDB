"""
tests/unit/test_normalizer.py

Unit tests for app.ingestion.normalizer.normalize().

normalize() is a pure function (no IO), so every test is synchronous.

Coverage
--------
  - Unicode NFC normalization
  - Null bytes and non-printable control characters stripped
  - CRLF and bare-CR converted to LF
  - Leading/trailing whitespace stripped per-line and overall
  - Intra-line whitespace collapsed to single space
  - Runs of 3+ blank lines collapsed to one blank line
  - Empty / whitespace-only input returns empty string
"""
from __future__ import annotations

import pytest

from app.ingestion.normalizer import normalize


class TestNormalizeBasic:
    def test_plain_text_unchanged_modulo_strip(self) -> None:
        result = normalize("Hello world.")
        assert result == "Hello world."

    def test_leading_trailing_whitespace_stripped(self) -> None:
        assert normalize("  hello  ") == "hello"

    def test_empty_string_returns_empty(self) -> None:
        assert normalize("") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        assert normalize("   \n\t\n   ") == ""


class TestNormalizeLineEndings:
    def test_crlf_converted_to_lf(self) -> None:
        result = normalize("line one\r\nline two")
        assert "\r" not in result
        assert result == "line one\nline two"

    def test_bare_cr_converted_to_lf(self) -> None:
        result = normalize("line one\rline two")
        assert "\r" not in result
        assert result == "line one\nline two"

    def test_mixed_line_endings_unified(self) -> None:
        result = normalize("a\r\nb\rc\nd")
        assert result == "a\nb\nc\nd"


class TestNormalizeControlCharacters:
    def test_null_bytes_removed(self) -> None:
        result = normalize("hello\x00world")
        assert "\x00" not in result
        assert result == "helloworld"

    def test_non_printable_control_chars_removed(self) -> None:
        # \x01–\x08 are stripped; \x09 (tab) and \x0a (newline) are kept.
        result = normalize("hello\x01\x02\x07world")
        assert result == "helloworld"

    def test_tab_kept_but_collapsed(self) -> None:
        # Tab is kept but collapsed into a single space with other whitespace.
        result = normalize("hello\t\tworld")
        assert result == "hello world"


class TestNormalizeIntraLineWhitespace:
    def test_multiple_spaces_collapsed(self) -> None:
        assert normalize("hello   world") == "hello world"

    def test_leading_spaces_per_line_stripped(self) -> None:
        result = normalize("  line one\n  line two")
        assert result == "line one\nline two"

    def test_trailing_spaces_per_line_stripped(self) -> None:
        result = normalize("line one   \nline two   ")
        assert result == "line one\nline two"


class TestNormalizeBlankLines:
    def test_single_blank_line_preserved(self) -> None:
        result = normalize("para one\n\npara two")
        assert result == "para one\n\npara two"

    def test_three_blank_lines_collapsed_to_one(self) -> None:
        result = normalize("para one\n\n\n\npara two")
        # 3+ consecutive newlines → 2 (one blank line)
        assert "\n\n\n" not in result
        assert "para one" in result
        assert "para two" in result

    def test_five_blank_lines_collapsed(self) -> None:
        result = normalize("a\n\n\n\n\n\nb")
        assert result.count("\n") <= 2


class TestNormalizeUnicode:
    def test_nfc_normalization_applied(self) -> None:
        # "é" can be encoded as precomposed (NFC) or decomposed (NFD).
        # After normalize(), both should produce the same NFC string.
        import unicodedata
        nfd = unicodedata.normalize("NFD", "café")
        nfc = unicodedata.normalize("NFC", "café")
        assert normalize(nfd) == normalize(nfc)

    def test_unicode_text_preserved(self) -> None:
        result = normalize("Yen: ¥1000, Euro: €500.")
        assert "¥" in result
        assert "€" in result
