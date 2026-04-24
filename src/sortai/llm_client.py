"""LM Studio client — model lifecycle management and chat completions."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from types import TracebackType
from typing import Optional

from openai import OpenAI


class LMStudioClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        prompts_dir: Path = Path("prompts"),
        temperature: float = 0.2,
        max_tokens: int = 2048,
        reasoning: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.prompts_dir = Path(prompts_dir)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self._openai = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="lm-studio",
        )

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def is_model_loaded(self) -> bool:
        """Return True if the model is already loaded in LM Studio."""
        models = self._openai.models.list()
        return any(m.id == self.model_name for m in models.data)

    def load_model(self) -> None:
        """POST /api/v1/models/load — no-op if the model is already loaded."""
        if not self.is_model_loaded():
            self._post_v1("models/load", {"model": self.model_name}, timeout=300)

    def unload_model(self) -> None:
        """POST /api/v1/models/unload to release the model from GPU memory."""
        self._post_v1("models/unload", {"instance_id": self.model_name}, timeout=60)

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Single-turn chat completion; no conversation history kept."""
        payload: dict = {
            "model": self.model_name,
            "input": prompt,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        if system:
            payload["system_prompt"] = system

        response = self._post_v1("chat", payload, timeout=300)
        for item in response.get("output", []):
            if item.get("type") == "message":
                return item.get("content", "")
        return ""

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    def load_prompt(self, name: str) -> str:
        """Return the contents of prompts/{name}.md."""
        return (self.prompts_dir / f"{name}.md").read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Context manager — auto load/unload
    # ------------------------------------------------------------------

    def __enter__(self) -> "LMStudioClient":
        self.load_model()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.unload_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, api_path: str, payload: dict, timeout: int = 60) -> dict:
        """POST JSON payload to an arbitrary LM Studio API path."""
        url = f"{self.base_url}/{api_path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"LM Studio API error {exc.code} on POST /{api_path}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach LM Studio at {self.base_url}.\n"
                "Make sure the local server is running:\n"
                "  1. Open LM Studio and click the Developer tab in the left sidebar.\n"
                "  2. Click the toggle next to 'Status: Stopped' to start the server."
            ) from exc

    def _post_v0(self, endpoint: str, payload: dict, timeout: int = 60) -> dict:
        return self._post(f"api/v0/{endpoint}", payload, timeout)

    def _post_v1(self, endpoint: str, payload: dict, timeout: int = 60) -> dict:
        return self._post(f"api/v1/{endpoint}", payload, timeout)
