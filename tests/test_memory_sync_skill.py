"""Tests for the standalone memory-sync skill helper."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SKILL_SCRIPT = Path(__file__).resolve().parent.parent / "memory-sync" / "scripts" / "memory_sync.py"


def load_module():
    spec = importlib.util.spec_from_file_location("memory_sync_skill", SKILL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_init_sync_and_check(tmp_path):
    mod = load_module()
    repo = tmp_path / "repo"
    repo.mkdir()

    mod.init_repo(repo)

    shared = repo / "memory" / "project.md"
    agents = repo / "AGENTS.md"
    claude = repo / "CLAUDE.md"

    assert shared.exists()
    assert agents.exists()
    assert claude.exists()
    assert "@AGENTS.md" in claude.read_text(encoding="utf-8")
    assert mod.check_repo(repo) == []


def test_check_detects_drift(tmp_path):
    mod = load_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    mod.init_repo(repo)

    (repo / "AGENTS.md").write_text("drifted\n", encoding="utf-8")
    assert mod.check_repo(repo) == ["AGENTS.md"]


def test_claude_overlay_is_appended(tmp_path):
    mod = load_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    mod.init_repo(repo)

    overlay = repo / "memory" / "claude.md"
    overlay.write_text("## Claude Code\n\n- host specific note\n", encoding="utf-8")
    mod.sync_repo(repo)

    assert "host specific note" in (repo / "CLAUDE.md").read_text(encoding="utf-8")
