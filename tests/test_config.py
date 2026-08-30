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


def test_reset_to_defaults_preserves_api_key(tmp_path):
    from app.config import Config, DEFAULT_CONFIG
    path = tmp_path / "config.json"
    cfg = Config(path)
    cfg.set("ai", "deepseek", "api_key", "sk-keep-me")
    cfg.set("hotkey", "key", "f8")
    cfg.set("subtitle", "font_size", 30)
    cfg.reset_to_defaults()
    assert cfg.get("ai", "deepseek", "api_key") == "sk-keep-me"
    assert cfg.get("hotkey", "key") == DEFAULT_CONFIG["hotkey"]["key"]
    assert cfg.get("subtitle", "font_size") == DEFAULT_CONFIG["subtitle"]["font_size"]
    # 存檔後重新載入也一致
    cfg2 = Config(path)
    assert cfg2.get("ai", "deepseek", "api_key") == "sk-keep-me"
    assert cfg2.get("hotkey", "key") == DEFAULT_CONFIG["hotkey"]["key"]


def test_system_captions_defaults(tmp_path):
    from app.config import Config
    cfg = Config(tmp_path / "config.json")
    assert cfg.get("system_captions", "enabled") is False
    assert cfg.get("system_captions", "hotkey_key") == "f11"
    assert cfg.get("system_captions", "engine") == "nllb-600m"
    assert cfg.get("system_captions", "language") == ""
    assert cfg.get("system_captions", "compute_device") == "auto"
    assert cfg.get("system_captions", "segment_silence_ms") == 400


def test_reset_restores_system_captions(tmp_path):
    from app.config import Config
    cfg = Config(tmp_path / "config.json")
    cfg.set("system_captions", "enabled", True)
    cfg.set("system_captions", "engine", "opus-mt")
    cfg.reset_to_defaults()
    assert cfg.get("system_captions", "enabled") is False
    assert cfg.get("system_captions", "engine") == "nllb-600m"
