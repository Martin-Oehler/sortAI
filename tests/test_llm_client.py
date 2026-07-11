"""Tests for sortai.llm_client.LMStudioClient."""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sortai.config import Config, LMStudioConfig
from sortai.llm_client import LMStudioClient


BASE_URL = "http://localhost:1234"
MODEL = "test-model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(tmp_path: Path, ttl: int | None = None) -> LMStudioClient:
    """Return a client that points at a real tmp prompts directory."""
    with patch("sortai.llm_client.OpenAI"):
        client = LMStudioClient(
            base_url=BASE_URL,
            model_name=MODEL,
            prompts_dir=tmp_path,
            temperature=0.3,
            max_tokens=512,
            ttl=ttl,
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
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_load_model_posts_correct_url_and_payload(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        # First call: GET /api/v1/models (is_model_loaded check) → model not present
        # Second call: POST /api/v1/models/load
        get_resp = _fake_urlopen_response(b'{"models": []}')
        post_resp = _fake_urlopen_response(b"{}")

        with patch(
            "sortai.llm_client.urllib.request.urlopen", side_effect=[get_resp, post_resp]
        ) as mock_open:
            client.load_model()

        assert mock_open.call_count == 2
        req = mock_open.call_args[0][0]
        assert req.full_url == f"{BASE_URL}/api/v1/models/load"
        assert req.get_method() == "POST"
        assert json.loads(req.data) == {"model": MODEL}
        assert req.get_header("Content-type") == "application/json"

    def test_load_model_is_noop_when_ttl_set(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, ttl=300)
        with patch("sortai.llm_client.urllib.request.urlopen") as mock_open:
            client.load_model()
        mock_open.assert_not_called()

    def test_load_model_uses_300s_timeout(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        mock_resp = _fake_urlopen_response(b"{}")

        with patch("sortai.llm_client.urllib.request.urlopen", return_value=mock_resp) as mock_open:
            client.load_model()

        # timeout is passed as a keyword argument to urlopen
        assert mock_open.call_args.kwargs["timeout"] == 300

    def test_http_error_raises_runtime_error(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        http_error = urllib.error.HTTPError(
            url=f"{BASE_URL}/api/v1/models/load",
            code=503,
            msg="Service Unavailable",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b"no model loaded"),
        )
        # GET /api/v1/models succeeds (model not loaded); POST /api/v1/models/load fails
        get_resp = _fake_urlopen_response(b'{"models": []}')

        with patch(
            "sortai.llm_client.urllib.request.urlopen", side_effect=[get_resp, http_error]
        ):
            with pytest.raises(RuntimeError) as exc_info:
                client.load_model()

        assert "503" in str(exc_info.value)
        assert "models/load" in str(exc_info.value)

    def test_http_error_message_includes_body(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)

        http_error = urllib.error.HTTPError(
            url=f"{BASE_URL}/api/v1/models/load",
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
# from_config factory
# ---------------------------------------------------------------------------

class TestFromConfig:
    def _make_config(self, tmp_path: Path, **lms_overrides) -> Config:
        lms = LMStudioConfig(
            base_url="http://example.local:9999",
            model="factory-model",
            temperature=0.7,
            max_tokens=1234,
            context_length=8192,
            model_ttl=120,
        )
        for key, value in lms_overrides.items():
            setattr(lms, key, value)
        return Config(
            inbox=tmp_path / "inbox",
            archive=tmp_path / "archive",
            prompts_dir=tmp_path / "prompts",
            lm_studio=lms,
        )

    def test_fields_match_config_values(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)

        with patch("sortai.llm_client.OpenAI"):
            client = LMStudioClient.from_config(cfg)

        assert client.base_url == "http://example.local:9999"
        assert client.model_name == "factory-model"
        assert client.prompts_dir == tmp_path / "prompts"
        assert client.temperature == 0.7
        assert client.max_tokens == 1234
        assert client.context_length == 8192
        assert client.ttl == 120

    def test_none_optionals_pass_through(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path, context_length=None, model_ttl=None)

        with patch("sortai.llm_client.OpenAI"):
            client = LMStudioClient.from_config(cfg)

        assert client.context_length is None
        assert client.ttl is None

    def test_returns_lm_studio_client_instance(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)

        with patch("sortai.llm_client.OpenAI") as MockOpenAI:
            client = LMStudioClient.from_config(cfg)

        assert isinstance(client, LMStudioClient)
        MockOpenAI.assert_called_once_with(
            base_url="http://example.local:9999/v1",
            api_key="lm-studio",
        )


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------

class TestClientInit:
    def test_openai_initialized_with_correct_base_url(self, tmp_path: Path) -> None:
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
# complete_structured() TTL
# ---------------------------------------------------------------------------

class TestCompleteStructuredTTL:
    _SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}

    def test_complete_structured_includes_ttl_in_extra_body_when_set(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, ttl=60)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"answer": "yes"}'
        client._openai.chat.completions.create = MagicMock(return_value=mock_response)

        client.complete_structured("prompt", self._SCHEMA)

        call_kwargs = client._openai.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("extra_body") == {"ttl": 60}

    def test_complete_structured_omits_extra_body_when_ttl_none(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, ttl=None)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"answer": "yes"}'
        client._openai.chat.completions.create = MagicMock(return_value=mock_response)

        client.complete_structured("prompt", self._SCHEMA)

        call_kwargs = client._openai.chat.completions.create.call_args.kwargs
        assert "extra_body" not in call_kwargs
