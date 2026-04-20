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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.prompts_dir = Path(prompts_dir)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._openai = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="lm-studio",
        )

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """POST /api/v0/models/load to ask LM Studio to load the model."""
        self._post_v0("models/load", {"identifier": self.model_name}, timeout=300)

    def unload_model(self) -> None:
        """POST /api/v0/models/unload to release the model from GPU memory."""
        self._post_v0("models/unload", {"identifier": self.model_name}, timeout=60)

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Single-turn chat completion; no conversation history kept."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

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

    def _post_v0(self, endpoint: str, payload: dict, timeout: int = 60) -> dict:
        url = f"{self.base_url}/api/v0/{endpoint}"
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
                f"LM Studio API error {exc.code} on POST /{endpoint}: {body}"
            ) from exc
