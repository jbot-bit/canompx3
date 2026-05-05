"""Tests for the standalone ~/.canompx-ask/ask.py CLI.

Loads the standalone script as a module via importlib.util so we don't
require it to be on sys.path. All filesystem state goes through tmp_path
to avoid touching the real install at ~/.canompx-ask/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest


ASK_SOURCE = Path.home() / ".canompx-ask" / "ask.py"
INSTALL_SOURCE = Path.home() / ".canompx-ask" / "install_ask.py"


def _load_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ask_mod(monkeypatch, tmp_path):
    if not ASK_SOURCE.exists():
        pytest.skip(f"standalone ask.py not installed at {ASK_SOURCE}")
    mod = _load_module(ASK_SOURCE, "_ask_standalone")
    install = tmp_path / ".canompx-ask"
    install.mkdir()
    monkeypatch.setattr(mod, "INSTALL_ROOT", install)
    monkeypatch.setattr(mod, "CONFIG_PATH", install / "config.toml")
    monkeypatch.setattr(mod, "MEMORY_DIR", install / "memory")
    monkeypatch.setattr(mod, "MEMORY_PATH", install / "memory" / "facts.jsonl")
    monkeypatch.setattr(mod, "CACHE_DIR", install / "cache")
    monkeypatch.setattr(mod, "MODELS_CACHE", install / "cache" / "openrouter_models.json")
    monkeypatch.setattr(mod, "HISTORY_DIR", install / "cache" / "history")
    return mod


# ---------- config ----------


def test_config_load_chain(ask_mod, tmp_path, monkeypatch):
    # Move cwd to a clean dir so _load_dotenv_chain doesn't pick up the real
    # repo's .env. Defaults only (no toml, no env, no .env in install root).
    clean = tmp_path / "clean"
    clean.mkdir()
    monkeypatch.chdir(clean)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ASK_MODEL", raising=False)
    monkeypatch.delenv("CANOMPX3_AI_CHAT_MODEL", raising=False)
    cfg = ask_mod._load_config()
    assert cfg["defaults"]["model"] == "deepseek/deepseek-chat"
    assert cfg["openrouter"]["api_key"] == ""

    # toml override.
    ask_mod.CONFIG_PATH.write_text(
        '[openrouter]\napi_key = "sk-or-fromfile"\n[defaults]\nmodel = "z-ai/glm-4"\n',
        encoding="utf-8",
    )
    cfg = ask_mod._load_config()
    assert cfg["openrouter"]["api_key"] == "sk-or-fromfile"
    assert cfg["defaults"]["model"] == "z-ai/glm-4"

    # env wins.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fromenv")
    monkeypatch.setenv("ASK_MODEL", "openai/gpt-4o-mini")
    cfg = ask_mod._load_config()
    assert cfg["openrouter"]["api_key"] == "sk-or-fromenv"
    assert cfg["defaults"]["model"] == "openai/gpt-4o-mini"


def test_dotenv_alt_name_open_router_api_key(ask_mod, tmp_path, monkeypatch):
    """ask should accept OPEN_ROUTER_API_KEY (with underscore) from a project .env."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)

    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".env").write_text(
        "# comment\nOPEN_ROUTER_API_KEY=sk-or-fromdotenv\nOTHER=ignored\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    cfg = ask_mod._load_config()
    assert cfg["openrouter"]["api_key"] == "sk-or-fromdotenv"


def test_dotenv_install_root_loaded(ask_mod, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    (ask_mod.INSTALL_ROOT / ".env").write_text("OPENROUTER_API_KEY=sk-or-installroot\n", encoding="utf-8")
    cfg = ask_mod._load_config()
    assert cfg["openrouter"]["api_key"] == "sk-or-installroot"


def test_shell_env_wins_over_dotenv(ask_mod, tmp_path, monkeypatch):
    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".env").write_text("OPENROUTER_API_KEY=sk-or-fromdotenv\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fromshell")
    cfg = ask_mod._load_config()
    assert cfg["openrouter"]["api_key"] == "sk-or-fromshell"


# ---------- memory ----------


def test_memory_roundtrip(ask_mod):
    fact = ask_mod._memory_append("user prefers ruff over black", tags=["always_on"])
    assert fact["id"].startswith("f-")
    assert "always_on" in fact["tags"]

    loaded = ask_mod._memory_load()
    assert len(loaded) == 1
    assert loaded[0]["text"] == "user prefers ruff over black"

    f2 = ask_mod._memory_append("user is in Brisbane", tags=[])
    assert len(ask_mod._memory_load()) == 2

    # Delete by id-prefix (use the unique suffix to avoid matching both).
    n = ask_mod._memory_delete(f2["id"])
    assert n == 1
    remaining = ask_mod._memory_load()
    assert len(remaining) == 1
    assert remaining[0]["id"] == fact["id"]


def test_recall_always_on_always_included(ask_mod):
    ask_mod._memory_append("ALWAYS prefer ruff over black", tags=["always_on"])
    ask_mod._memory_append("user likes mountain biking", tags=[])
    block = ask_mod._recall("how do I deploy to AWS?", top_n=5, max_chars=500)
    assert "ruff" in block  # always_on present
    assert "Known about user:" in block


def test_recall_token_overlap_ranking(ask_mod):
    ask_mod._memory_append("user knows kubernetes deployment well", tags=[])
    ask_mod._memory_append("user enjoys cooking pasta on weekends", tags=[])
    block = ask_mod._recall("how do I scale my kubernetes deployment?", top_n=1, max_chars=500)
    assert "kubernetes" in block
    assert "pasta" not in block


def test_recall_budget_cap(ask_mod):
    long_fact = "x" * 300
    for i in range(10):
        ask_mod._memory_append(f"{long_fact} #{i} kubernetes", tags=[])
    block = ask_mod._recall("kubernetes question", top_n=10, max_chars=500)
    assert len(block) <= 500


def test_no_memory_bypass_returns_unchanged_system(ask_mod):
    ask_mod._memory_append("user prefers ruff", tags=["always_on"])
    cfg = ask_mod._load_config()
    args = argparse.Namespace(no_memory=True, no_project=True)
    out = ask_mod._augment_system("BASE_SYS", "format my code", args, cfg)
    assert out == "BASE_SYS"


def test_recall_empty_when_no_facts(ask_mod):
    block = ask_mod._recall("anything", top_n=5, max_chars=500)
    assert block == ""


# ---------- project context ----------


def test_project_context_finds_claude_md_via_git_walk(ask_mod, tmp_path, monkeypatch):
    repo = tmp_path / "myproj"
    repo.mkdir()
    (repo / ".git").mkdir()  # marker only — no real git init needed
    (repo / "CLAUDE.md").write_text("# Project rules\n- Use ruff\n", encoding="utf-8")
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)

    cfg = ask_mod._load_config()
    out = ask_mod._project_context(cfg)
    assert out.startswith("Project context (from myproj/CLAUDE.md):")
    assert "Use ruff" in out


def test_project_context_skips_outside_repo(ask_mod, tmp_path, monkeypatch):
    nogit = tmp_path / "no-repo"
    nogit.mkdir()
    monkeypatch.chdir(nogit)
    cfg = ask_mod._load_config()
    assert ask_mod._project_context(cfg) == ""


def test_project_context_disabled_via_config(ask_mod, tmp_path, monkeypatch):
    repo = tmp_path / "p"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "CLAUDE.md").write_text("rules", encoding="utf-8")
    monkeypatch.chdir(repo)
    cfg = ask_mod._load_config()
    cfg["project_context"]["enabled"] = False
    assert ask_mod._project_context(cfg) == ""


def test_project_context_max_chars_truncates(ask_mod, tmp_path, monkeypatch):
    repo = tmp_path / "p"
    repo.mkdir()
    (repo / ".git").mkdir()
    big = "Y" * 5000
    (repo / "CLAUDE.md").write_text(big, encoding="utf-8")
    monkeypatch.chdir(repo)
    cfg = ask_mod._load_config()
    cfg["project_context"]["max_chars"] = 100
    out = ask_mod._project_context(cfg)
    assert "Y" * 100 in out
    assert len(out) < 250  # excerpt + header, but < 5000


def test_project_context_prefers_first_listed_file(ask_mod, tmp_path, monkeypatch):
    repo = tmp_path / "p"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "CLAUDE.md").write_text("CLAUDE_CONTENT", encoding="utf-8")
    (repo / "AGENTS.md").write_text("AGENTS_CONTENT", encoding="utf-8")
    monkeypatch.chdir(repo)
    cfg = ask_mod._load_config()
    out = ask_mod._project_context(cfg)
    assert "CLAUDE_CONTENT" in out
    assert "AGENTS_CONTENT" not in out


def test_project_context_freshness_no_caching(ask_mod, tmp_path, monkeypatch):
    """CLAUDE.md must be re-read every call so live edits surface immediately."""
    repo = tmp_path / "p"
    repo.mkdir()
    (repo / ".git").mkdir()
    target = repo / "CLAUDE.md"
    target.write_text("v1", encoding="utf-8")
    monkeypatch.chdir(repo)
    cfg = ask_mod._load_config()
    out1 = ask_mod._project_context(cfg)
    assert "v1" in out1
    target.write_text("v2-updated", encoding="utf-8")
    out2 = ask_mod._project_context(cfg)
    assert "v2-updated" in out2


# ---------- CLI parser ----------


def test_parser_accepts_new_flags(ask_mod):
    p = ask_mod.build_parser()
    args = p.parse_args(["--remember", "x", "--always-on", "--tag", "vim"])
    assert args.remember == "x"
    assert args.always_on is True
    assert args.tag == ["vim"]
    args = p.parse_args(["--no-memory", "--no-project", "hello"])
    assert args.no_memory is True
    assert args.no_project is True


# ---------- installer ----------


@pytest.mark.skipif(not INSTALL_SOURCE.exists(), reason="installer not present")
def test_install_idempotent(tmp_path, monkeypatch):
    """Running the installer twice preserves config + memory and refreshes ask.py."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")  # bypasses prompt

    install_mod = _load_module(INSTALL_SOURCE, "_install_ask")
    monkeypatch.setattr(install_mod, "INSTALL", tmp_path / ".canompx-ask")

    # First run — set up directories + config without venv (skip _create_venv to keep test fast).
    monkeypatch.setattr(install_mod, "_create_venv", lambda *a, **kw: Path(sys.executable))
    monkeypatch.setattr(install_mod, "_pip_install", lambda *a, **kw: None)
    monkeypatch.setattr(install_mod, "_setx_path_windows", lambda *a, **kw: None)
    monkeypatch.setattr(install_mod, "_bashrc_path_inject", lambda *a, **kw: None)

    rc = install_mod.main([])
    assert rc == 0
    config = tmp_path / ".canompx-ask" / "config.toml"
    assert config.exists()
    config_first = config.read_text(encoding="utf-8")

    # Pre-populate a fact and re-run — must not be wiped.
    facts = tmp_path / ".canompx-ask" / "memory" / "facts.jsonl"
    facts.write_text('{"id":"f-test","ts":"x","text":"persist","tags":[]}\n', encoding="utf-8")

    rc2 = install_mod.main([])
    assert rc2 == 0
    assert config.read_text(encoding="utf-8") == config_first  # config preserved
    assert "persist" in facts.read_text(encoding="utf-8")  # memory preserved


def test_path_dedup_is_case_insensitive_on_windows(ask_mod, monkeypatch):
    if not INSTALL_SOURCE.exists():
        pytest.skip("installer not present")
    install_mod = _load_module(INSTALL_SOURCE, "_install_ask_pathdedup")
    monkeypatch.setattr(install_mod, "IS_WINDOWS", True)
    target = r"C:\Users\joe\.canompx-ask\bin"
    monkeypatch.setenv("PATH", r"C:\foo;C:\Users\Joe\.canompx-ask\BIN;C:\bar")
    assert install_mod._path_already_has(target) is True
    monkeypatch.setenv("PATH", r"C:\foo;C:\bar")
    assert install_mod._path_already_has(target) is False
