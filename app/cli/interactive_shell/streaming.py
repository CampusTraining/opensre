"""Live token streaming for interactive-shell LLM responses.

Drives a Rich.Live region while the model emits text chunks, re-rendering the
buffer as Markdown each chunk so formatting (bold, lists, code spans) appears
progressively — same feel as Claude CLI. Falls back to a drain-and-return on
non-terminal consoles so captured logs stay clean.
"""

from __future__ import annotations

from collections.abc import Iterator

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

from app.cli.interactive_shell.theme import TERMINAL_ACCENT_BOLD

# Match loaders.py spinner so streaming and non-streaming surfaces look
# identical. Duplicated rather than imported because loaders.py keeps these
# constants file-local by convention.
_SPINNER_NAME = "dots12"
_SPINNER_COLOR = "orange1"
_SPINNER_LABEL = "thinking"


def stream_to_console(
    console: Console,
    *,
    label: str,
    chunks: Iterator[str],
) -> str:
    """Render a streaming LLM response live, return the accumulated text.

    On a terminal console, prints the ``label`` header, then opens a
    ``rich.live.Live`` region that starts on a "thinking…" spinner and
    swaps to a Markdown render of the accumulated buffer once chunks
    start arriving. Each chunk re-renders the buffer in place — the user
    sees text appear progressively with formatting (bold, lists, code
    spans) applied live. The final content stays visible on screen; no
    clear-and-reprint at finalize.

    On a non-terminal console (CI, captured stdout, piped output) the
    stream is drained silently and returned as a single string. No Live
    region, no spinner artifacts in captured logs.

    If the stream raises mid-flight, whatever was accumulated remains on
    screen as the exception propagates so the caller can surface an
    error label below the partial response.
    """
    if not console.is_terminal:
        return "".join(chunks)

    buffer: list[str] = []
    spinner = Spinner(
        _SPINNER_NAME,
        text=Text(f"{_SPINNER_LABEL}…", style=f"bold {_SPINNER_COLOR}"),
        style=f"bold {_SPINNER_COLOR}",
    )

    console.print()
    console.print(f"[{TERMINAL_ACCENT_BOLD}]{label}:[/]")

    try:
        with Live(spinner, console=console, refresh_per_second=10, transient=False) as live:
            for chunk in chunks:
                if not chunk:
                    continue
                buffer.append(chunk)
                live.update(Markdown("".join(buffer)))
            # If no chunks arrived, replace the frozen spinner with empty
            # content so it doesn't get stuck on screen.
            if not buffer:
                live.update(Text(""))
    finally:
        console.print()

    return "".join(buffer)


__all__ = ["stream_to_console"]
