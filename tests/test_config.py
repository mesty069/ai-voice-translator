import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config, DEFAULT_CONFIG


def test_creates_default_config(tmp_path):
    path = tmp_path / "config.json"
    cfg = Config(path)
    assert path.exists()
    assert cfg.get("ai", "provider") == "deepseek"
    assert cfg.get("hotkey", "key") == "f9"


def test_merges_partial_user_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"hotkey": {"key": "f8"}}), encoding="utf-8")
    cfg = Config(path)
    assert cfg.get("hotkey", "key") == "f8"
    assert cfg.get("hotkey", "type") == "keyboard"  # 未覆寫的用預設值
    assert cfg.get("ai", "deepseek", "model") == "deepseek-chat"


def test_set_persists(tmp_path):
    path = tmp_path / "config.json"
    cfg = Config(path)
    cfg.set("ai", "deepseek", "api_key", "sk-test")
    reloaded = Config(path)
    assert reloaded.get("ai", "deepseek", "api_key") == "sk-test"


def test_corrupt_config_falls_back_to_default(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    cfg = Config(path)
    assert cfg.data == DEFAULT_CONFIG
