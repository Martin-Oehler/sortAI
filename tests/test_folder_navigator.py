"""Tests for sortai.folder_navigator."""
from __future__ import annotations

from pathlib import Path

import pytest

from sortai.folder_navigator import is_leaf, list_children


class TestListChildren:
    def test_returns_sorted_directory_names(self, tmp_path: Path) -> None:
        """list_children returns sub-directory names in sorted order."""
        (tmp_path / "zebra").mkdir()
        (tmp_path / "alpha").mkdir()
        (tmp_path / "middle").mkdir()

        result = list_children(tmp_path)

        assert result == ["alpha", "middle", "zebra"]

    def test_ignores_files(self, tmp_path: Path) -> None:
        """list_children does not include files, only directories."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.pdf").write_bytes(b"%PDF")

        result = list_children(tmp_path)

        assert result == ["subdir"]

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """list_children on a directory with no children returns []."""
        result = list_children(tmp_path)
        assert result == []

    def test_only_files_returns_empty_list(self, tmp_path: Path) -> None:
        """list_children returns [] when the directory contains only files."""
        (tmp_path / "file.txt").write_text("content")
        result = list_children(tmp_path)
        assert result == []


class TestIsLeaf:
    def test_true_when_no_subdirectories(self, tmp_path: Path) -> None:
        """is_leaf returns True for a directory that has no sub-directories."""
        # Even with files present it should still be a leaf.
        (tmp_path / "document.pdf").write_bytes(b"%PDF")

        assert is_leaf(tmp_path) is True

    def test_true_for_empty_directory(self, tmp_path: Path) -> None:
        """is_leaf returns True for a completely empty directory."""
        assert is_leaf(tmp_path) is True

    def test_false_when_subdirectory_exists(self, tmp_path: Path) -> None:
        """is_leaf returns False when at least one sub-directory is present."""
        (tmp_path / "subdir").mkdir()

        assert is_leaf(tmp_path) is False

    def test_false_when_multiple_subdirectories_exist(self, tmp_path: Path) -> None:
        """is_leaf returns False when multiple sub-directories are present."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()

        assert is_leaf(tmp_path) is False
