import json
import copy
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

DEFAULT_CONFIG = {
    "ai": {
        "provider": "deepseek",
        "deepseek": {
            "api_key": "",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
        },
    },
    "stt": {
        "model_size": "small",
        "device": "cpu",
        "compute_type": "int8",
    },
    "hotkey": {
        "type": "keyboard",
        "key": "f9",
    },
    "output": {
        "auto_copy": False,
        "tts_enabled": False,
        "tts_device": "default",
    },
    "mute_other_apps": True,
    "ui": {
        "theme": "auto",
        "theme_color": "#0078d4",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    def __init__(self, path: Path = CONFIG_PATH):
        self.path = path
        self.data = copy.deepcopy(DEFAULT_CONFIG)
        self.load()

    def load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
                self.data = _deep_merge(DEFAULT_CONFIG, user_data)
            except (json.JSONDecodeError, OSError):
                self.data = copy.deepcopy(DEFAULT_CONFIG)
        else:
            self.save()

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get(self, *keys, default=None):
        node = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set(self, *keys_and_value):
        *keys, value = keys_and_value
        node = self.data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
        self.save()
