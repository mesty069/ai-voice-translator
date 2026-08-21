from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from qfluentwidgets import (
    BodyLabel,
    ComboBox,
    LineEdit,
    PasswordLineEdit,
    ScrollArea,
    StrongBodyLabel,
    SwitchButton,
    setTheme,
    setThemeColor,
    Theme,
)

from ..config import Config
from ..core import tts

KEYBOARD_KEYS = [f"f{i}" for i in range(1, 13)] + [
    "ctrl_r", "alt_r", "shift_r", "caps_lock", "scroll_lock", "pause"]
MOUSE_KEYS = ["middle", "x1", "x2"]

STT_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

THEMES = [("跟隨系統", "auto"), ("淺色", "light"), ("深色", "dark")]
THEME_COLORS = [
    ("Windows 藍", "#0078d4"),
    ("綠", "#107c10"),
    ("紫", "#8764b8"),
    ("橘", "#ca5010"),
    ("紅", "#d13438"),
]


class SettingsInterface(ScrollArea):
    hotkey_changed = Signal()
    stt_model_changed = Signal(str)

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsInterface")
        self.config = config
        self._loading = True

        self.view = QWidget(self)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()

        layout = QVBoxLayout(self.view)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(10)

        # ---- AI ----
        layout.addWidget(StrongBodyLabel("AI 翻譯（DeepSeek）", self.view))
        self.api_key_edit = PasswordLineEdit(self.view)
        self.api_key_edit.setPlaceholderText("DeepSeek API Key")
        self._add_row(layout, "API Key", self.api_key_edit)
        self.base_url_edit = LineEdit(self.view)
        self._add_row(layout, "Base URL", self.base_url_edit)
        self.model_edit = LineEdit(self.view)
        self._add_row(layout, "模型", self.model_edit)

        # ---- STT ----
        layout.addSpacing(8)
        layout.addWidget(StrongBodyLabel("語音辨識（faster-whisper）", self.view))
        self.stt_combo = ComboBox(self.view)
        self.stt_combo.addItems(STT_MODELS)
        self._add_row(layout, "模型大小（越大越準、越慢）", self.stt_combo)

        # ---- 熱鍵 ----
        layout.addSpacing(8)
        layout.addWidget(StrongBodyLabel("熱鍵（按住說話）", self.view))
        self.hotkey_type_combo = ComboBox(self.view)
        self.hotkey_type_combo.addItems(["鍵盤", "滑鼠"])
        self._add_row(layout, "裝置", self.hotkey_type_combo)
        self.hotkey_key_combo = ComboBox(self.view)
        self._add_row(layout, "按鍵", self.hotkey_key_combo)

        # ---- 輸出 ----
        layout.addSpacing(8)
        layout.addWidget(StrongBodyLabel("輸出", self.view))
        self.auto_copy_switch = SwitchButton(self.view)
        self._add_row(layout, "翻譯完成自動複製英文到剪貼簿", self.auto_copy_switch)
        self.tts_switch = SwitchButton(self.view)
        self._add_row(layout, "用語音唸出英文（TTS）", self.tts_switch)
        self.tts_device_combo = ComboBox(self.view)
        self.tts_device_combo.addItem("default")
        try:
            self.tts_device_combo.addItems(tts.list_output_devices())
        except Exception:
            pass
        self._add_row(layout, "TTS 播放裝置", self.tts_device_combo)

        # ---- 麥克風 ----
        layout.addSpacing(8)
        layout.addWidget(StrongBodyLabel("麥克風保護", self.view))
        self.mute_switch = SwitchButton(self.view)
        self._add_row(layout, "錄音時暫時靜音其他程式的麥克風", self.mute_switch)

        # ---- 外觀 ----
        layout.addSpacing(8)
        layout.addWidget(StrongBodyLabel("外觀", self.view))
        self.theme_combo = ComboBox(self.view)
        self.theme_combo.addItems([name for name, _ in THEMES])
        self._add_row(layout, "主題", self.theme_combo)
        self.color_combo = ComboBox(self.view)
        self.color_combo.addItems([name for name, _ in THEME_COLORS])
        self._add_row(layout, "主題色", self.color_combo)

        layout.addStretch(1)

        self._load_from_config()
        self._connect_signals()
        self._loading = False

    def _add_row(self, layout: QVBoxLayout, label: str, widget: QWidget):
        row = QHBoxLayout()
        row.addWidget(BodyLabel(label, self.view))
        row.addStretch(1)
        widget.setMinimumWidth(240)
        row.addWidget(widget, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(row)

    # ---- 載入 ----

    def _load_from_config(self):
        cfg = self.config
        self.api_key_edit.setText(cfg.get("ai", "deepseek", "api_key", default=""))
        self.base_url_edit.setText(
            cfg.get("ai", "deepseek", "base_url", default="https://api.deepseek.com"))
        self.model_edit.setText(
            cfg.get("ai", "deepseek", "model", default="deepseek-chat"))

        stt_model = cfg.get("stt", "model_size", default="small")
        if stt_model in STT_MODELS:
            self.stt_combo.setCurrentIndex(STT_MODELS.index(stt_model))

        hotkey_type = cfg.get("hotkey", "type", default="keyboard")
        self.hotkey_type_combo.setCurrentIndex(0 if hotkey_type == "keyboard" else 1)
        self._populate_keys(hotkey_type)
        key = cfg.get("hotkey", "key", default="f9")
        keys = KEYBOARD_KEYS if hotkey_type == "keyboard" else MOUSE_KEYS
        if key in keys:
            self.hotkey_key_combo.setCurrentIndex(keys.index(key))

        self.auto_copy_switch.setChecked(
            cfg.get("output", "auto_copy", default=False))
        self.tts_switch.setChecked(cfg.get("output", "tts_enabled", default=False))
        device = cfg.get("output", "tts_device", default="default")
        for i in range(self.tts_device_combo.count()):
            if self.tts_device_combo.itemText(i) == device:
                self.tts_device_combo.setCurrentIndex(i)
                break

        self.mute_switch.setChecked(cfg.get("mute_other_apps", default=True))

        theme = cfg.get("ui", "theme", default="auto")
        for i, (_, value) in enumerate(THEMES):
            if value == theme:
                self.theme_combo.setCurrentIndex(i)
        color = cfg.get("ui", "theme_color", default="#0078d4")
        for i, (_, value) in enumerate(THEME_COLORS):
            if value.lower() == color.lower():
                self.color_combo.setCurrentIndex(i)

    def _populate_keys(self, hotkey_type: str):
        keys = KEYBOARD_KEYS if hotkey_type == "keyboard" else MOUSE_KEYS
        self.hotkey_key_combo.clear()
        self.hotkey_key_combo.addItems(keys)

    # ---- 變更 → 存檔 ----

    def _connect_signals(self):
        self.api_key_edit.editingFinished.connect(
            lambda: self._save("ai", "deepseek", "api_key",
                               self.api_key_edit.text().strip()))
        self.base_url_edit.editingFinished.connect(
            lambda: self._save("ai", "deepseek", "base_url",
                               self.base_url_edit.text().strip()))
        self.model_edit.editingFinished.connect(
            lambda: self._save("ai", "deepseek", "model",
                               self.model_edit.text().strip()))

        self.stt_combo.currentIndexChanged.connect(self._on_stt_changed)
        self.hotkey_type_combo.currentIndexChanged.connect(self._on_hotkey_type_changed)
        self.hotkey_key_combo.currentIndexChanged.connect(self._on_hotkey_key_changed)

        self.auto_copy_switch.checkedChanged.connect(
            lambda checked: self._save("output", "auto_copy", checked))
        self.tts_switch.checkedChanged.connect(
            lambda checked: self._save("output", "tts_enabled", checked))
        self.tts_device_combo.currentIndexChanged.connect(
            lambda _: self._save("output", "tts_device",
                                 self.tts_device_combo.currentText()))
        self.mute_switch.checkedChanged.connect(
            lambda checked: self._save("mute_other_apps", checked))

        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self.color_combo.currentIndexChanged.connect(self._on_color_changed)

    def _save(self, *keys_and_value):
        if self._loading:
            return
        self.config.set(*keys_and_value)

    def _on_stt_changed(self, index: int):
        if self._loading:
            return
        model = STT_MODELS[index]
        self.config.set("stt", "model_size", model)
        self.stt_model_changed.emit(model)

    def _on_hotkey_type_changed(self, index: int):
        if self._loading:
            return
        hotkey_type = "keyboard" if index == 0 else "mouse"
        self._loading = True
        self._populate_keys(hotkey_type)
        self._loading = False
        default_key = "f9" if hotkey_type == "keyboard" else "x2"
        keys = KEYBOARD_KEYS if hotkey_type == "keyboard" else MOUSE_KEYS
        self.hotkey_key_combo.setCurrentIndex(keys.index(default_key))
        self.config.set("hotkey", "type", hotkey_type)
        self.config.set("hotkey", "key", default_key)
        self.hotkey_changed.emit()

    def _on_hotkey_key_changed(self, index: int):
        if self._loading or index < 0:
            return
        self.config.set("hotkey", "key", self.hotkey_key_combo.currentText())
        self.hotkey_changed.emit()

    def _on_theme_changed(self, index: int):
        if self._loading:
            return
        value = THEMES[index][1]
        self.config.set("ui", "theme", value)
        setTheme({"auto": Theme.AUTO, "light": Theme.LIGHT,
                  "dark": Theme.DARK}[value])

    def _on_color_changed(self, index: int):
        if self._loading:
            return
        value = THEME_COLORS[index][1]
        self.config.set("ui", "theme_color", value)
        setThemeColor(QColor(value))
