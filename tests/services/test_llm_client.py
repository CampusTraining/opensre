from __future__ import annotations

from app.services import llm_client


class _FakeAnthropicMessages:
    def create(self, **_kwargs):
        raise AssertionError("unexpected network call in unit test")


class _FakeAnthropic:
    last_api_key: str | None = None

    def __init__(self, *, api_key: str, timeout: float) -> None:
        _FakeAnthropic.last_api_key = api_key
        self.timeout = timeout
        self.messages = _FakeAnthropicMessages()


class _FakeOpenAICompletions:
    def create(self, **_kwargs):
        raise AssertionError("unexpected network call in unit test")


class _FakeOpenAIChat:
    def __init__(self) -> None:
        self.completions = _FakeOpenAICompletions()


class _FakeOpenAI:
    last_api_key: str | None = None
    last_base_url: str | None = None
    last_default_headers: dict[str, str] | None = None

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: float,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        _FakeOpenAI.last_api_key = api_key
        _FakeOpenAI.last_base_url = base_url
        _FakeOpenAI.last_default_headers = default_headers
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers
        self.chat = _FakeOpenAIChat()


def test_openai_llm_client_reads_secure_local_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "resolve_llm_api_key",
        lambda env_var: "stored-openai-key" if env_var == "OPENAI_API_KEY" else "",
    )
    monkeypatch.setattr(llm_client, "OpenAI", _FakeOpenAI)

    client = llm_client.OpenAILLMClient(model="gpt-5.4")
    client._ensure_client()

    assert _FakeOpenAI.last_api_key == "stored-openai-key"


def test_anthropic_llm_client_reads_secure_local_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "resolve_llm_api_key",
        lambda env_var: "stored-anthropic-key" if env_var == "ANTHROPIC_API_KEY" else "",
    )
    monkeypatch.setattr(llm_client, "Anthropic", _FakeAnthropic)

    client = llm_client.LLMClient(model="claude-opus-4")
    client._ensure_client()

    assert _FakeAnthropic.last_api_key == "stored-anthropic-key"


def test_minimax_llm_client_reads_api_key_and_base_url(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "resolve_llm_api_key",
        lambda env_var: "minimax-test-key" if env_var == "MINIMAX_API_KEY" else "",
    )
    monkeypatch.setattr(llm_client, "OpenAI", _FakeOpenAI)

    client = llm_client.OpenAILLMClient(
        model="MiniMax-M2.7",
        base_url="https://api.minimax.io/v1",
        api_key_env="MINIMAX_API_KEY",
        temperature=1.0,
    )
    client._ensure_client()

    assert _FakeOpenAI.last_api_key == "minimax-test-key"
    assert _FakeOpenAI.last_base_url == "https://api.minimax.io/v1"


def test_minimax_llm_client_temperature_is_set(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "resolve_llm_api_key",
        lambda env_var: "minimax-test-key" if env_var == "MINIMAX_API_KEY" else "",
    )
    monkeypatch.setattr(llm_client, "OpenAI", _FakeOpenAI)

    client = llm_client.OpenAILLMClient(
        model="MiniMax-M2.7",
        base_url="https://api.minimax.io/v1",
        api_key_env="MINIMAX_API_KEY",
        temperature=1.0,
    )
    assert client._temperature == 1.0


# ---------------------------------------------------------------------------
# LLMClient.invoke / invoke_stream — kwargs builder + streaming behavior
# ---------------------------------------------------------------------------


def _make_capturing_anthropic(
    *,
    response_text: str = "",
    chunks: list[str] | None = None,
):
    """Build a fake Anthropic class that captures kwargs and returns canned data.

    Distinct from the module-level ``_FakeAnthropic`` (which raises on any
    API call to guard ``_ensure_client`` tests). Closure-scoped so each test
    gets a fresh capture dict — no class-level state to reset between tests.
    """
    state: dict = {"kwargs": None}
    stream_chunks = list(chunks or [])

    class _Block:
        def __init__(self, text: str) -> None:
            self.type = "text"
            self.text = text

    class _Response:
        def __init__(self, text: str) -> None:
            self.content = [_Block(text)]

    class _StreamCM:
        def __init__(self) -> None:
            self.text_stream = iter(stream_chunks)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _Messages:
        def create(self, **kwargs):
            state["kwargs"] = kwargs
            return _Response(response_text)

        def stream(self, **kwargs):
            state["kwargs"] = kwargs
            return _StreamCM()

    class _Anthropic:
        def __init__(self, *, api_key: str, timeout: float) -> None:
            self.api_key = api_key
            self.timeout = timeout
            self.messages = _Messages()

    return _Anthropic, state


def test_anthropic_invoke_forwards_built_kwargs_to_messages_create(monkeypatch) -> None:
    """Refactored invoke() still sends model, max_tokens, and messages to the SDK."""
    fake, captured = _make_capturing_anthropic(response_text="hello")
    monkeypatch.setattr(llm_client, "resolve_llm_api_key", lambda _env: "k")
    monkeypatch.setattr(llm_client, "Anthropic", fake)

    client = llm_client.LLMClient(model="claude-test", max_tokens=64)
    response = client.invoke("hi")

    assert response.content == "hello"
    assert captured["kwargs"]["model"] == "claude-test"
    assert captured["kwargs"]["max_tokens"] == 64
    assert captured["kwargs"]["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_invoke_stream_yields_text_stream_chunks(monkeypatch) -> None:
    """invoke_stream() routes through the same builder and yields SDK chunks in order."""
    fake, captured = _make_capturing_anthropic(chunks=["Hel", "lo, ", "world"])
    monkeypatch.setattr(llm_client, "resolve_llm_api_key", lambda _env: "k")
    monkeypatch.setattr(llm_client, "Anthropic", fake)

    client = llm_client.LLMClient(model="claude-test", max_tokens=64)
    chunks = list(client.invoke_stream("hi"))

    assert chunks == ["Hel", "lo, ", "world"]
    assert captured["kwargs"]["model"] == "claude-test"
    assert captured["kwargs"]["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_invoke_stream_applies_guardrails_to_input(monkeypatch) -> None:
    """The shared kwargs builder runs guardrail redaction before the stream opens."""
    fake, captured = _make_capturing_anthropic(chunks=["ok"])
    monkeypatch.setattr(llm_client, "resolve_llm_api_key", lambda _env: "k")
    monkeypatch.setattr(llm_client, "Anthropic", fake)

    class _RedactingEngine:
        is_active = True

        def apply(self, content: str) -> str:
            return content.replace("secret", "[REDACTED]")

    import app.guardrails.engine as engine_module

    monkeypatch.setattr(engine_module, "get_guardrail_engine", lambda: _RedactingEngine())

    client = llm_client.LLMClient(model="claude-test")
    list(client.invoke_stream("share my secret"))

    assert captured["kwargs"]["messages"][0]["content"] == "share my [REDACTED]"


# ---------------------------------------------------------------------------
# OpenAILLMClient.invoke / invoke_stream — kwargs builder + streaming behavior
# ---------------------------------------------------------------------------


def _make_capturing_openai(
    *,
    response_text: str = "",
    chunk_contents: list[str | None] | None = None,
):
    """Build a fake OpenAI class that captures kwargs and returns canned data.

    ``chunk_contents`` accepts ``None`` entries to simulate empty deltas the
    real SDK emits during keep-alive — invoke_stream must skip those.
    Closure-scoped so each test gets a fresh capture dict.
    """
    state: dict = {"kwargs": None}
    raw_chunks = list(chunk_contents or [])

    class _Delta:
        def __init__(self, content: str | None) -> None:
            self.content = content

    class _Choice:
        def __init__(self, *, delta_content: str | None = None, message_content: str = "") -> None:
            self.delta = _Delta(delta_content)
            self.message = type("_Msg", (), {"content": message_content})()

    class _Response:
        def __init__(self, message_content: str) -> None:
            self.choices = [_Choice(message_content=message_content)]

    class _StreamChunk:
        def __init__(self, content: str | None) -> None:
            self.choices = [_Choice(delta_content=content)] if content is not None else []

    class _Completions:
        def create(self, **kwargs):
            state["kwargs"] = kwargs
            if kwargs.get("stream"):
                return iter(_StreamChunk(c) for c in raw_chunks)
            return _Response(response_text)

    class _Chat:
        def __init__(self) -> None:
            self.completions = _Completions()

    class _OpenAI:
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str | None = None,
            timeout: float,
            default_headers: dict[str, str] | None = None,
        ) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.timeout = timeout
            self.default_headers = default_headers
            self.chat = _Chat()

    return _OpenAI, state


def test_openai_invoke_forwards_built_kwargs_to_chat_completions_create(monkeypatch) -> None:
    """Refactored invoke() still sends model, max_tokens, and messages to the SDK."""
    fake, captured = _make_capturing_openai(response_text="hello")
    monkeypatch.setattr(llm_client, "resolve_llm_api_key", lambda _env: "k")
    monkeypatch.setattr(llm_client, "OpenAI", fake)

    client = llm_client.OpenAILLMClient(model="gpt-test", max_tokens=64)
    response = client.invoke("hi")

    assert response.content == "hello"
    assert captured["kwargs"]["model"] == "gpt-test"
    assert captured["kwargs"]["max_tokens"] == 64
    assert captured["kwargs"]["messages"] == [{"role": "user", "content": "hi"}]
    assert "stream" not in captured["kwargs"]


def test_openai_invoke_stream_yields_delta_content_chunks(monkeypatch) -> None:
    """invoke_stream() routes through the same builder and yields delta.content in order."""
    fake, captured = _make_capturing_openai(chunk_contents=["Hel", "lo, ", "world"])
    monkeypatch.setattr(llm_client, "resolve_llm_api_key", lambda _env: "k")
    monkeypatch.setattr(llm_client, "OpenAI", fake)

    client = llm_client.OpenAILLMClient(model="gpt-test", max_tokens=64)
    chunks = list(client.invoke_stream("hi"))

    assert chunks == ["Hel", "lo, ", "world"]
    assert captured["kwargs"]["stream"] is True
    assert captured["kwargs"]["model"] == "gpt-test"
    assert captured["kwargs"]["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_invoke_stream_skips_empty_deltas_and_choiceless_chunks(monkeypatch) -> None:
    """OpenAI keep-alive frames have empty delta or no choices — those must not be yielded."""
    fake, _ = _make_capturing_openai(chunk_contents=["Hi", None, "", " there", None])
    monkeypatch.setattr(llm_client, "resolve_llm_api_key", lambda _env: "k")
    monkeypatch.setattr(llm_client, "OpenAI", fake)

    client = llm_client.OpenAILLMClient(model="gpt-test")
    chunks = list(client.invoke_stream("hi"))

    assert chunks == ["Hi", " there"]
