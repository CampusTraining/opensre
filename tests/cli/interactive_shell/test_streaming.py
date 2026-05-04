"""Tests for the shared live-streaming renderer used by interactive-shell handlers."""

from __future__ import annotations

import io
import re
from collections.abc import Iterator

import pytest
from rich.console import Console

from app.cli.interactive_shell.streaming import stream_to_console


def _strip_ansi(text: str) -> str:
    """Drop ANSI escapes so assertions check the visible output."""
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


def _tty_console() -> tuple[Console, io.StringIO]:
    """Build a Console that thinks it is a terminal so Rich.Live actually renders."""
    buf = io.StringIO()
    return (
        Console(file=buf, force_terminal=True, color_system=None, width=80, highlight=False),
        buf,
    )


def _non_tty_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, color_system=None, width=80), buf


def _yield_chunks(chunks: list[str]) -> Iterator[str]:
    yield from chunks


class TestNonTtyFallback:
    """On a non-terminal console the helper drains silently and returns the full text."""

    def test_drains_stream_and_returns_full_text(self) -> None:
        console, buf = _non_tty_console()
        result = stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks(["Hel", "lo, ", "world"]),
        )

        assert result == "Hello, world"
        # No Live-region escape sequences leak into captured output.
        assert "thinking" not in buf.getvalue()
        assert "assistant:" not in buf.getvalue()


class TestTtyLiveRender:
    """On a terminal console the response renders live and the final text stays visible."""

    def test_renders_label_and_streamed_content_as_markdown(self) -> None:
        console, buf = _tty_console()
        result = stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks(["Run **opensre", " investigate** to start."]),
        )

        output = _strip_ansi(buf.getvalue())
        assert result == "Run **opensre investigate** to start."
        # Header is pinned above the live region.
        assert "assistant:" in output
        # Markdown is rendered live; the literal ** delimiters must not survive.
        assert "**opensre" not in output
        assert "opensre investigate" in output

    def test_returns_empty_string_when_stream_is_empty(self) -> None:
        """An empty stream must not leave a frozen spinner on screen."""
        console, buf = _tty_console()
        result = stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks([]),
        )

        assert result == ""
        # Header still printed, but no thinking-spinner residue at finalize.
        assert "assistant:" in _strip_ansi(buf.getvalue())


class TestMidStreamError:
    """Errors inside the stream propagate while the partial buffer stays on screen."""

    def test_exception_propagates_with_partial_visible(self) -> None:
        def _broken_stream() -> Iterator[str]:
            yield "partial "
            yield "answer"
            raise RuntimeError("upstream 503")

        console, buf = _tty_console()

        with pytest.raises(RuntimeError, match="upstream 503"):
            stream_to_console(
                console,
                label="assistant",
                chunks=_broken_stream(),
            )

        # The partial response was rendered before the exception propagated,
        # so the caller can surface an error label below it.
        output = _strip_ansi(buf.getvalue())
        assert "partial answer" in output
