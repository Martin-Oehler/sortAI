"""Tests for sortai.llm_client.LMStudioClient."""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from sortai.llm_client import LLMResponse, LMStudioClient


BASE_URL = "http://localhost:1234"
MODEL = "test-model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(tmp_path: Path) -> LMStudioClient:
    """Return a client that points at a real tmp prompts directory."""
    with patch("sortai.llm_client.OpenAI"):
        client = LMStudioClient(
            base_url=BASE_URL,
            model_name=MODEL,
            prompts_dir=tmp_path,
            temperature=0.3,
            max_tokens=512,
        )
    return client


def _fake_urlopen_response(body: bytes = b"{}"):
    """Return a mock context-manager response for urllib.request.urlopen."""
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# _post_v0 / load_model / unload_model
# ---------------------------------------------------------------------------

class TestPostV0:
    def test_load_model_posts_correct_url_and_payload(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        mock_resp = _fake_urlopen_response(b"{}")

        with patch("sortai.llm_client.urllib.request.urlopen", return_value=mock_resp) as mock_open:
            client.load_model()

        assert mock_open.call_count == 1
        req = mock_open.call_args[0][0]
        assert req.full_url == f"{BASE_URL}/api/v1/models/load"
        assert req.get_method() == "POST"
        assert json.loads(req.data) == {"model": MODEL}
        assert req.get_header("Content-type") == "application/json"

    def test_load_model_uses_300s_timeout(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        mock_resp = _fake_urlopen_response(b"{}")

        with patch("sortai.llm_client.urllib.request.urlopen", return_value=mock_resp) as mock_open:
            client.load_model()

        # timeout is passed as a keyword argument to urlopen
        assert mock_open.call_args.kwargs["timeout"] == 300

    def test_unload_model_posts_correct_url_and_payload(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        mock_resp = _fake_urlopen_response(b"{}")

        with patch("sortai.llm_client.urllib.request.urlopen", return_value=mock_resp) as mock_open:
            client.unload_model()

        req = mock_open.call_args[0][0]
        assert req.full_url == f"{BASE_URL}/api/v1/models/unload"
        assert req.get_method() == "POST"
        assert json.loads(req.data) == {"instance_id": MODEL}

    def test_unload_model_uses_60s_timeout(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        mock_resp = _fake_urlopen_response(b"{}")

        with patch("sortai.llm_client.urllib.request.urlopen", return_value=mock_resp) as mock_open:
            client.unload_model()

        assert mock_open.call_args.kwargs["timeout"] == 60

    def test_http_error_raises_runtime_error(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        http_error = urllib.error.HTTPError(
            url=f"{BASE_URL}/api/v0/models/load",
            code=503,
            msg="Service Unavailable",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b"no model loaded"),
        )

        with patch("sortai.llm_client.urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(RuntimeError) as exc_info:
                client.load_model()

        assert "503" in str(exc_info.value)
        assert "models/load" in str(exc_info.value)

    def test_http_error_message_includes_body(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        http_error = urllib.error.HTTPError(
            url=f"{BASE_URL}/api/v0/models/load",
            code=422,
            msg="Unprocessable Entity",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b"bad identifier"),
        )

        with patch("sortai.llm_client.urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(RuntimeError, match="bad identifier"):
                client.load_model()

    def test_connection_refused_gives_helpful_message(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        url_error = urllib.error.URLError(ConnectionRefusedError(10061, "Connection refused"))

        with patch("sortai.llm_client.urllib.request.urlopen", side_effect=url_error):
            with pytest.raises(RuntimeError) as exc_info:
                client.load_model()

        msg = str(exc_info.value)
        assert BASE_URL in msg
        assert "Developer" in msg

    def test_base_url_trailing_slash_is_stripped(self, tmp_path: Path) -> None:
        """Trailing slash on base_url must not produce double slashes in the URL."""
        with patch("sortai.llm_client.OpenAI"):
            client = LMStudioClient(
                base_url="http://localhost:1234/",
                model_name=MODEL,
                prompts_dir=tmp_path,
            )
        mock_resp = _fake_urlopen_response(b"{}")

        with patch("sortai.llm_client.urllib.request.urlopen", return_value=mock_resp) as mock_open:
            client.load_model()

        req = mock_open.call_args[0][0]
        assert "//" not in req.full_url.replace("http://", "")


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------

class TestComplete:
    def _post_response(self, content: str | None = "hello", reasoning: str | None = None) -> dict:
        output = []
        if content is not None:
            output.append({"type": "message", "content": content})
        if reasoning is not None:
            output.append({"type": "reasoning", "content": reasoning})
        return {"output": output}

    def test_complete_returns_llmresponse_with_content(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        with patch.object(client, "_post_v1", return_value=self._post_response("The answer is 42.")):
            result = client.complete("What is the answer?")
        assert isinstance(result, LLMResponse)
        assert result.content == "The answer is 42."

    def test_complete_returns_empty_reasoning_when_absent(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        with patch.object(client, "_post_v1", return_value=self._post_response("hello")):
            result = client.complete("test")
        assert result.reasoning == ""

    def test_complete_captures_reasoning_token(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        response = self._post_response("answer", reasoning="I thought about it carefully")
        with patch.object(client, "_post_v1", return_value=response):
            result = client.complete("test")
        assert result.content == "answer"
        assert result.reasoning == "I thought about it carefully"

    def test_complete_returns_empty_content_when_no_message_output(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        with patch.object(client, "_post_v1", return_value={"output": []}):
            result = client.complete("test")
        assert result.content == ""
        assert result.reasoning == ""

    def test_complete_with_system_includes_system_prompt_in_payload(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        captured: list[dict] = []
        def capture(endpoint, payload, **kwargs):
            captured.append(payload)
            return self._post_response("ok")
        with patch.object(client, "_post_v1", side_effect=capture):
            client.complete("Hi", system="You are helpful.")
        assert captured[0]["system_prompt"] == "You are helpful."
        assert captured[0]["input"] == "Hi"

    def test_complete_without_system_omits_system_prompt(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        captured: list[dict] = []
        def capture(endpoint, payload, **kwargs):
            captured.append(payload)
            return self._post_response("ok")
        with patch.object(client, "_post_v1", side_effect=capture):
            client.complete("ping")
        assert "system_prompt" not in captured[0]

    def test_complete_payload_includes_model_temperature_tokens(self, tmp_path: Path) -> None:
        with patch("sortai.llm_client.OpenAI"):
            client = LMStudioClient(
                base_url=BASE_URL,
                model_name=MODEL,
                prompts_dir=tmp_path,
                temperature=0.7,
                max_tokens=1024,
            )
        captured: list[dict] = []
        def capture(endpoint, payload, **kwargs):
            captured.append(payload)
            return self._post_response("ok")
        with patch.object(client, "_post_v1", side_effect=capture):
            client.complete("test")
        assert captured[0]["model"] == MODEL
        assert captured[0]["temperature"] == 0.7
        assert captured[0]["max_output_tokens"] == 1024

    def test_complete_openai_initialized_with_correct_base_url(self, tmp_path: Path) -> None:
        with patch("sortai.llm_client.OpenAI") as MockOpenAI:
            LMStudioClient(
                base_url=BASE_URL,
                model_name=MODEL,
                prompts_dir=tmp_path,
            )

        MockOpenAI.assert_called_once_with(
            base_url=f"{BASE_URL}/v1",
            api_key="lm-studio",
        )


# ---------------------------------------------------------------------------
# load_prompt()
# ---------------------------------------------------------------------------

class TestLoadPrompt:
    def test_reads_correct_file(self, tmp_path: Path) -> None:
        (tmp_path / "classify.md").write_text("Classify the document.", encoding="utf-8")
        client = _make_client(tmp_path)

        result = client.load_prompt("classify")

        assert result == "Classify the document."

    def test_reads_nested_prompt_name(self, tmp_path: Path) -> None:
        (tmp_path / "summary.md").write_text("Summarize this.", encoding="utf-8")
        client = _make_client(tmp_path)

        assert client.load_prompt("summary") == "Summarize this."

    def test_missing_prompt_raises_file_not_found(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        with pytest.raises(FileNotFoundError):
            client.load_prompt("nonexistent")


# ---------------------------------------------------------------------------
# Context manager (__enter__ / __exit__)
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_enter_calls_load_model(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.load_model = MagicMock()
        client.unload_model = MagicMock()

        with client:
            pass

        client.load_model.assert_called_once()

    def test_exit_calls_unload_model(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.load_model = MagicMock()
        client.unload_model = MagicMock()

        with client:
            pass

        client.unload_model.assert_called_once()

    def test_enter_returns_client(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.load_model = MagicMock()
        client.unload_model = MagicMock()

        with client as c:
            assert c is client

    def test_unload_called_even_if_exception_raised(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.load_model = MagicMock()
        client.unload_model = MagicMock()

        with pytest.raises(ValueError):
            with client:
                raise ValueError("boom")

        client.unload_model.assert_called_once()

    def test_load_called_before_body_executes(self, tmp_path: Path) -> None:
        call_order: list[str] = []
        client = _make_client(tmp_path)
        client.load_model = MagicMock(side_effect=lambda: call_order.append("load"))
        client.unload_model = MagicMock(side_effect=lambda: call_order.append("unload"))

        with client:
            call_order.append("body")

        assert call_order == ["load", "body", "unload"]
