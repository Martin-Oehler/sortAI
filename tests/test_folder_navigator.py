"""Tests for sortai.folder_navigator."""
from __future__ import annotations

from pathlib import Path

import pytest

from sortai.folder_navigator import FolderInfo, is_leaf, list_children, list_children_with_info


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


class TestListChildrenWithInfo:
    def test_returns_folder_info_for_each_child(self, tmp_path: Path) -> None:
        """list_children_with_info returns one FolderInfo per sub-directory."""
        (tmp_path / "alpha").mkdir()
        (tmp_path / "beta").mkdir()

        result = list_children_with_info(tmp_path)

        assert [i.name for i in result] == ["alpha", "beta"]

    def test_subfolders_populated(self, tmp_path: Path) -> None:
        """FolderInfo.subfolders lists the child's own sub-directories."""
        child = tmp_path / "parent"
        child.mkdir()
        (child / "sub1").mkdir()
        (child / "sub2").mkdir()

        result = list_children_with_info(tmp_path)

        assert result[0].subfolders == ["sub1", "sub2"]

    def test_subfolders_truncated_to_preview_count(self, tmp_path: Path) -> None:
        """Subfolder list is capped at subfolder_preview_count."""
        child = tmp_path / "parent"
        child.mkdir()
        for i in range(10):
            (child / f"sub{i:02d}").mkdir()

        result = list_children_with_info(tmp_path, subfolder_preview_count=3)

        assert len(result[0].subfolders) == 3

    def test_empty_child_has_empty_subfolders(self, tmp_path: Path) -> None:
        """A leaf folder results in subfolders=[]."""
        (tmp_path / "leaf").mkdir()

        result = list_children_with_info(tmp_path)

        assert result[0].subfolders == []
        assert result[0].description is None

    def test_description_file_read(self, tmp_path: Path) -> None:
        """Description file contents are read and stripped."""
        child = tmp_path / "invoices"
        child.mkdir()
        (child / "folder-description.md").write_text("  Supplier invoices by year.  \n", encoding="utf-8")

        result = list_children_with_info(tmp_path)

        assert result[0].description == "Supplier invoices by year."

    def test_missing_description_file_gives_none(self, tmp_path: Path) -> None:
        """description is None when no description file is present."""
        (tmp_path / "contracts").mkdir()

        result = list_children_with_info(tmp_path)

        assert result[0].description is None

    def test_custom_description_filename(self, tmp_path: Path) -> None:
        """A custom description_filename is used for lookup."""
        child = tmp_path / "docs"
        child.mkdir()
        (child / "my-desc.txt").write_text("Custom description", encoding="utf-8")

        result = list_children_with_info(tmp_path, description_filename="my-desc.txt")

        assert result[0].description == "Custom description"

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """No children → empty list."""
        result = list_children_with_info(tmp_path)
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
