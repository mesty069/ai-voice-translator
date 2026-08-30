import json
import copy
import sys
from pathlib import Path

# 打包（PyInstaller）後以 exe 所在資料夾為基準；開發時以專案根目錄為基準
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = BASE_DIR / "config.json"

# 工作列釘選/群組識別，程式視窗與捷徑（開始功能表、開機啟動）須一致
APP_ID = "CharlieCheng.AIVoiceZh2En"

# (whisper 語言代碼, 顯示名稱)；代碼同時用於語音辨識與 TTS 語音挑選
LANGUAGES = [
    ("zh", "中文"),
    ("en", "英文"),
    ("ja", "日文"),
    ("ko", "韓文"),
    ("es", "西班牙文"),
    ("fr", "法文"),
    ("de", "德文"),
    ("vi", "越南文"),
    ("th", "泰文"),
    ("ru", "俄文"),
]
# 給 AI 提示詞用的語言名稱（中文要指明繁體）
_PROMPT_NAMES = {"zh": "繁體中文"}


def lang_display(code: str) -> str:
    return dict(LANGUAGES).get(code, code)


def lang_prompt_name(code: str) -> str:
    return _PROMPT_NAMES.get(code, lang_display(code))

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
        # cuda 在沒有 GPU 的機器上會自動退回 CPU（stt.py 的 fallback）
        # int8_float16：顯存需求約砍半、載入與推論更快，精度損失極小
        "model_size": "large-v3",
        "device": "cuda",
        "compute_type": "int8_float16",
    },
    "language": {
        "source": "zh",   # 母語（你說的語言）
        "target": "en",   # 目標翻譯語言
    },
    "hotkey": {
        "type": "keyboard",
        "key": "f9",
    },
    "replay_hotkey": {
        "type": "keyboard",
        "key": "",
    },
    "grammar": {
        "enabled": False,
        "hotkey_type": "keyboard",
        "hotkey_key": "f10",
    },
    "output": {
        "auto_copy": False,
        "tts_enabled": True,
        "tts_device": "default",
        "tts_rate": 200,
    },
    "recording": {
        "device": "default",
        "isolate_other_devices": True,
        # 隔離後等其他軟體跟隨裝置切換的緩衝（毫秒），之後才開錄、亮紅色
        "isolation_settle_ms": 700,
    },
    "subtitle": {
        "duration_seconds": 10,
        "font_size": 21,
        "opacity": 100,
        "bg_color": "#121212",
        "font_color": "#ffffff",
        "font_weight": "bold",
        "font_family": "",
    },
    "float_input": {
        "enabled": False,
        "opacity": 100,
    },
    "system_captions": {
        "enabled": False,
        "hotkey_type": "keyboard",
        "hotkey_key": "f11",
        "device": "default",
        "language": "",            # 空字串＝跟隨目標語言
        "engine": "nllb-600m",     # nllb-600m | nllb-1.3b | opus-mt
        "compute_device": "auto",  # auto | cpu
        "font_size": 20,
        "opacity": 100,
        "bg_color": "#0d2b33",
        "segment_silence_ms": 400,
        "max_segment_sec": 4,
    },
    "ui": {
        "theme": "auto",
        "theme_color": "#0078d4",
        "auto_bubble": True,
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

    def reset_to_defaults(self, preserve=(("ai", "deepseek", "api_key"),)):
        """全部恢復預設值；preserve 列出的 key 路徑保留原值（如 API key）。"""
        kept = [(keys, self.get(*keys)) for keys in preserve]
        self.data = copy.deepcopy(DEFAULT_CONFIG)
        for keys, value in kept:
            if value is not None:
                node = self.data
                for key in keys[:-1]:
                    node = node.setdefault(key, {})
                node[keys[-1]] = value
        self.save()
