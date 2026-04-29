"""Configuration loading and validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LMStudioConfig:
    base_url: str = "http://localhost:1234"
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    reasoning: Optional[str] = None


@dataclass
class Config:
    inbox: Path = field(default_factory=lambda: Path("."))
    archive: Path = field(default_factory=lambda: Path("."))
    prompts_dir: Path = field(default_factory=lambda: Path("prompts"))
    log_file: Path = field(default_factory=lambda: Path("logs/sortai.jsonl"))
    dry_run: bool = False
    max_navigate_depth: int = 10
    folder_description_filename: str = "folder-description.md"
    subfolder_preview_count: int = 5
    lm_studio: LMStudioConfig = field(default_factory=LMStudioConfig)

    @classmethod
    def load(cls, path: Path) -> "Config":
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Copy config/config.example.toml to {path} and fill in your paths."
            )
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        lms_raw = raw.get("lm_studio", {})
        lms = LMStudioConfig(
            base_url=lms_raw.get("base_url", "http://localhost:1234"),
            model=lms_raw.get("model", ""),
            temperature=lms_raw.get("temperature", 0.2),
            max_tokens=lms_raw.get("max_tokens", 2048),
            reasoning=lms_raw.get("reasoning"),
        )

        cfg = cls(
            inbox=Path(raw["inbox"]),
            archive=Path(raw["archive"]),
            prompts_dir=Path(raw.get("prompts_dir", "prompts")),
            log_file=Path(raw.get("log_file", "logs/sortai.jsonl")),
            dry_run=raw.get("dry_run", False),
            max_navigate_depth=raw.get("max_navigate_depth", 10),
            folder_description_filename=raw.get("folder_description_filename", "folder-description.md"),
            subfolder_preview_count=raw.get("subfolder_preview_count", 5),
            lm_studio=lms,
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        errors = []
        if not self.inbox.exists():
            errors.append(f"inbox path does not exist: {self.inbox}")
        if not self.archive.exists():
            errors.append(f"archive path does not exist: {self.archive}")
        if not self.lm_studio.model:
            errors.append("lm_studio.model must be set")
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
