from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QWidget

from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ColorDialog,
    ComboBox,
    LineEdit,
    PasswordLineEdit,
    PushButton,
    ScrollArea,
    SpinBox,
    StrongBodyLabel,
    SubtitleLabel,
    SwitchButton,
    setTheme,
    setThemeColor,
    Theme,
)

from ..config import Config, LANGUAGES, lang_display
from ..core import tts
from ..core.hotkey import MOUSE_BUTTONS
from ..core.local_translate import ENGINE_LABELS
from ..core.recorder import DEFAULT_DEVICE, list_input_devices
from ..core.system_audio import SystemAudioCapture

STT_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

MOUSE_KEY_DISPLAY = {"middle": "滑鼠中鍵", "x1": "滑鼠側鍵 1", "x2": "滑鼠側鍵 2"}

SUBTITLE_WEIGHTS = [
    ("細", "light"), ("正常", "normal"), ("中等", "medium"),
    ("半粗", "demibold"), ("粗", "bold"), ("特粗", "black"),
]


def hotkey_display(hotkey_type: str, key_name: str) -> str:
    if hotkey_type == "mouse":
        return MOUSE_KEY_DISPLAY.get(key_name, key_name)
    return key_name.upper()


class _CaptureBridge(QObject):
    """pynput 監聽執行緒 → GUI 執行緒的橋樑。"""
    captured = Signal(str, str)   # hotkey_type, key_name
    cancelled = Signal()


class HotkeyCaptureDialog(QDialog):
    """按下任意鍵盤按鍵或滑鼠鍵（中鍵/側鍵）即設定為熱鍵，Esc 取消。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定錄音熱鍵")
        self.setModal(True)
        self.setFixedSize(380, 150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(8)
        layout.addWidget(SubtitleLabel("請按下任意按鍵或滑鼠鍵", self))
        layout.addWidget(CaptionLabel(
            "將設定為「按住說話」的熱鍵。\n"
            "滑鼠左右鍵不可使用；按 Esc 取消。", self))
        layout.addStretch(1)

        self.result_type = None
        self.result_key = None
        self._bridge = _CaptureBridge()
        self._bridge.captured.connect(self._on_captured)
        self._bridge.cancelled.connect(self.reject)

        from pynput import keyboard, mouse
        self._kb = keyboard.Listener(on_press=self._on_key)
        self._ms = mouse.Listener(on_click=self._on_click)
        self._kb.start()
        self._ms.start()

    # ---- pynput 執行緒 ----

    def _on_key(self, key):
        from pynput import keyboard
        if key == keyboard.Key.esc:
            self._bridge.cancelled.emit()
            return
        if isinstance(key, keyboard.Key):
            self._bridge.captured.emit("keyboard", key.name)
            return
        char = getattr(key, "char", None)
        if char and char.strip():
            self._bridge.captured.emit("keyboard", char.lower())

    def _on_click(self, x, y, button, pressed):
        if not pressed:
            return
        for name, btn in MOUSE_BUTTONS.items():
            if button == btn:
                self._bridge.captured.emit("mouse", name)
                return

    # ---- GUI 執行緒 ----

    def _on_captured(self, hotkey_type: str, key_name: str):
        self.result_type = hotkey_type
        self.result_key = key_name
        self.accept()

    def done(self, result):
        self._kb.stop()
        self._ms.stop()
        super().done(result)

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
    hotkey_capture_started = Signal()
    stt_model_changed = Signal(str)
    mic_device_changed = Signal()
    float_input_toggled = Signal(bool)
    overlay_opacity_changed = Signal()
    subtitle_style_changed = Signal()
    error_requested = Signal(str)
    settings_reset = Signal()
    languages_changed = Signal()
    system_captions_toggled = Signal(bool)
    system_captions_settings_changed = Signal()
    system_captions_pipeline_changed = Signal()  # 只有需要重啟擷取的設定才發

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

        # ---- 語言 ----
        layout.addWidget(StrongBodyLabel("語言", self.view))
        self.source_lang_combo = ComboBox(self.view)
        self.source_lang_combo.addItems([n for _, n in LANGUAGES])
        self._add_row(layout, "母語（你說的語言）", self.source_lang_combo)
        self.target_lang_combo = ComboBox(self.view)
        self.target_lang_combo.addItems([n for _, n in LANGUAGES])
        self._add_row(layout, "目標翻譯語言", self.target_lang_combo)

        # ---- AI ----
        layout.addSpacing(8)
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
        self.hotkey_button = PushButton("", self.view)
        self.hotkey_button.setToolTip("點一下，然後按任意按鍵或滑鼠鍵")
        self._add_row(layout, "錄音熱鍵（點擊後按下想用的鍵）", self.hotkey_button)
        replay_row = QWidget(self.view)
        replay_layout = QHBoxLayout(replay_row)
        replay_layout.setContentsMargins(0, 0, 0, 0)
        replay_layout.setSpacing(6)
        self.replay_hotkey_button = PushButton("", replay_row)
        self.replay_hotkey_button.setToolTip("點一下，然後按任意按鍵或滑鼠鍵")
        replay_layout.addWidget(self.replay_hotkey_button, stretch=1)
        self.replay_hotkey_clear = PushButton("清除", replay_row)
        self.replay_hotkey_clear.setFixedWidth(56)
        replay_layout.addWidget(self.replay_hotkey_clear)
        self._add_row(layout, "朗讀熱鍵（字幕顯示時重播英文語音）", replay_row)

        # ---- 英文文法檢查 ----
        layout.addSpacing(8)
        layout.addWidget(StrongBodyLabel("文法檢查（目標語言）", self.view))
        self.grammar_switch = SwitchButton(self.view)
        self._add_row(layout, "啟用（按住熱鍵說目標語言，AI 有錯才修正、沒錯不回覆）",
                      self.grammar_switch)
        self.grammar_hotkey_button = PushButton("", self.view)
        self.grammar_hotkey_button.setToolTip("點一下，然後按任意按鍵或滑鼠鍵")
        self._add_row(layout, "文法檢查熱鍵", self.grammar_hotkey_button)

        # ---- 系統聲音字幕 ----
        layout.addSpacing(8)
        layout.addWidget(StrongBodyLabel("系統聲音字幕", self.view))
        self.system_captions_switch = SwitchButton(self.view)
        self._add_row(layout, "啟用（把電腦播放的聲音即時翻成母語字幕）",
                      self.system_captions_switch)
        self.system_hotkey_button = PushButton("", self.view)
        self.system_hotkey_button.setToolTip("點一下，然後按任意按鍵或滑鼠鍵")
        self._add_row(layout, "系統字幕開關熱鍵", self.system_hotkey_button)
        self.system_device_combo = ComboBox(self.view)
        self.system_device_combo.addItem("系統預設")
        try:
            self.system_device_combo.addItems(
                SystemAudioCapture.list_output_devices())
        except Exception:
            pass
        self._add_row(layout, "擷取來源（哪個喇叭的聲音）",
                      self.system_device_combo)
        self.system_language_combo = ComboBox(self.view)
        self.system_language_combo.addItem("跟隨目標語言")
        self.system_language_combo.addItems([n for _, n in LANGUAGES])
        self._add_row(layout, "系統聲音的語言", self.system_language_combo)
        self.system_engine_combo = ComboBox(self.view)
        self.system_engine_combo.addItems([n for _, n in ENGINE_LABELS])
        self._add_row(layout, "本機翻譯模型", self.system_engine_combo)
        self.system_compute_combo = ComboBox(self.view)
        self.system_compute_combo.addItems(["自動（優先 GPU）", "只用 CPU"])
        self._add_row(layout, "翻譯運算裝置", self.system_compute_combo)
        self.system_rows_spin = SpinBox(self.view)
        self.system_rows_spin.setRange(1, 5)
        self.system_rows_spin.setSuffix(" 句")
        self._add_row(layout, "保留前幾句翻譯（它們會接在正在講這句的前面）",
                      self.system_rows_spin)
        self.system_font_spin = SpinBox(self.view)
        self.system_font_spin.setRange(12, 48)
        self.system_font_spin.setSuffix(" px")
        self._add_row(layout, "系統字幕字體大小", self.system_font_spin)
        self.system_opacity_spin = SpinBox(self.view)
        self.system_opacity_spin.setRange(30, 100)
        self.system_opacity_spin.setSingleStep(5)
        self.system_opacity_spin.setSuffix(" %")
        self._add_row(layout, "系統字幕透明度", self.system_opacity_spin)

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
        self.tts_rate_spin = SpinBox(self.view)
        self.tts_rate_spin.setRange(80, 350)
        self.tts_rate_spin.setSingleStep(10)
        self._add_row(layout, "朗讀語速（預設 200，越大越快）", self.tts_rate_spin)
        self.tts_volume_spin = SpinBox(self.view)
        self.tts_volume_spin.setRange(0, 100)
        self.tts_volume_spin.setSingleStep(5)
        self.tts_volume_spin.setSuffix(" %")
        self._add_row(layout, "朗讀音量（預設 100%）", self.tts_volume_spin)

        # ---- 麥克風 ----
        layout.addSpacing(8)
        layout.addWidget(StrongBodyLabel("麥克風", self.view))
        self.mic_device_combo = ComboBox(self.view)
        self.mic_device_combo.addItem("系統預設")
        try:
            self.mic_device_combo.addItems(list_input_devices())
        except Exception:
            pass
        self._add_row(layout, "錄音麥克風", self.mic_device_combo)
        self.isolate_switch = SwitchButton(self.view)
        self._add_row(layout, "錄音時隔離麥克風（其他軟體聽不到你說話）",
                      self.isolate_switch)

        # ---- 懸浮球字幕 ----
        layout.addSpacing(8)
        layout.addWidget(StrongBodyLabel("懸浮球字幕", self.view))
        self.subtitle_duration_spin = SpinBox(self.view)
        self.subtitle_duration_spin.setRange(3, 120)
        self.subtitle_duration_spin.setSuffix(" 秒")
        self._add_row(layout, "字幕顯示時間", self.subtitle_duration_spin)
        self.subtitle_font_spin = SpinBox(self.view)
        self.subtitle_font_spin.setRange(12, 48)
        self.subtitle_font_spin.setSuffix(" px")
        self._add_row(layout, "英文字體大小（中文按比例縮放）",
                      self.subtitle_font_spin)
        self.subtitle_opacity_spin = SpinBox(self.view)
        self.subtitle_opacity_spin.setRange(30, 100)
        self.subtitle_opacity_spin.setSingleStep(5)
        self.subtitle_opacity_spin.setSuffix(" %")
        self._add_row(layout, "字幕透明度", self.subtitle_opacity_spin)
        self.subtitle_bg_button = PushButton("", self.view)
        self._add_row(layout, "字幕底色", self.subtitle_bg_button)
        self.subtitle_color_button = PushButton("", self.view)
        self._add_row(layout, "字體顏色", self.subtitle_color_button)
        self.subtitle_weight_combo = ComboBox(self.view)
        self.subtitle_weight_combo.addItems([n for n, _ in SUBTITLE_WEIGHTS])
        self._add_row(layout, "字體粗細", self.subtitle_weight_combo)
        self.subtitle_family_combo = ComboBox(self.view)
        self.subtitle_family_combo.addItem("預設")
        try:
            from PySide6.QtGui import QFontDatabase
            self.subtitle_family_combo.addItems(QFontDatabase.families())
        except Exception:
            pass
        self._add_row(layout, "字體樣式", self.subtitle_family_combo)
        self.float_input_switch = SwitchButton(self.view)
        self._add_row(layout, "懸浮球模式顯示輸入框（也可長按懸浮球切換）",
                      self.float_input_switch)
        self.input_opacity_spin = SpinBox(self.view)
        self.input_opacity_spin.setRange(30, 100)
        self.input_opacity_spin.setSingleStep(5)
        self.input_opacity_spin.setSuffix(" %")
        self._add_row(layout, "輸入框透明度", self.input_opacity_spin)

        # ---- 外觀 ----
        layout.addSpacing(8)
        layout.addWidget(StrongBodyLabel("外觀", self.view))
        self.theme_combo = ComboBox(self.view)
        self.theme_combo.addItems([name for name, _ in THEMES])
        self._add_row(layout, "主題", self.theme_combo)
        self.color_combo = ComboBox(self.view)
        self.color_combo.addItems([name for name, _ in THEME_COLORS])
        self._add_row(layout, "主題色", self.color_combo)

        # ---- 一般 ----
        layout.addSpacing(8)
        layout.addWidget(StrongBodyLabel("一般", self.view))
        self.autostart_switch = SwitchButton(self.view)
        self._add_row(layout, "開機自動啟動（背景預先載入語音模型）",
                      self.autostart_switch)
        self.auto_bubble_switch = SwitchButton(self.view)
        self._add_row(layout, "視窗被切到背景時自動縮成懸浮球",
                      self.auto_bubble_switch)

        # ---- 恢復預設 ----
        layout.addSpacing(16)
        self.reset_button = PushButton("全部恢復預設值", self.view)
        self.reset_button.clicked.connect(self._on_reset_defaults)
        layout.addWidget(self.reset_button)

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
        codes = [c for c, _ in LANGUAGES]
        src = cfg.get("language", "source", default="zh")
        tgt = cfg.get("language", "target", default="en")
        self.source_lang_combo.setCurrentIndex(
            codes.index(src) if src in codes else 0)
        self.target_lang_combo.setCurrentIndex(
            codes.index(tgt) if tgt in codes else 1)
        self.api_key_edit.setText(cfg.get("ai", "deepseek", "api_key", default=""))
        self.base_url_edit.setText(
            cfg.get("ai", "deepseek", "base_url", default="https://api.deepseek.com"))
        self.model_edit.setText(
            cfg.get("ai", "deepseek", "model", default="deepseek-chat"))

        stt_model = cfg.get("stt", "model_size", default="small")
        if stt_model in STT_MODELS:
            self.stt_combo.setCurrentIndex(STT_MODELS.index(stt_model))

        self._refresh_hotkey_button()

        self.auto_copy_switch.setChecked(
            cfg.get("output", "auto_copy", default=False))
        self.tts_switch.setChecked(cfg.get("output", "tts_enabled", default=False))
        self.tts_device_combo.setCurrentIndex(0)
        device = cfg.get("output", "tts_device", default="default")
        for i in range(self.tts_device_combo.count()):
            if self.tts_device_combo.itemText(i) == device:
                self.tts_device_combo.setCurrentIndex(i)
                break
        self.tts_rate_spin.setValue(cfg.get("output", "tts_rate", default=200))
        self.tts_volume_spin.setValue(
            cfg.get("output", "tts_volume", default=100))

        self.mic_device_combo.setCurrentIndex(0)
        mic_device = cfg.get("recording", "device", default=DEFAULT_DEVICE)
        if mic_device != DEFAULT_DEVICE:
            for i in range(1, self.mic_device_combo.count()):
                if self.mic_device_combo.itemText(i) == mic_device:
                    self.mic_device_combo.setCurrentIndex(i)
                    break
        self.isolate_switch.setChecked(
            cfg.get("recording", "isolate_other_devices", default=True))

        self.subtitle_duration_spin.setValue(
            cfg.get("subtitle", "duration_seconds", default=10))
        self.subtitle_font_spin.setValue(
            cfg.get("subtitle", "font_size", default=21))
        self.float_input_switch.setChecked(
            cfg.get("float_input", "enabled", default=False))
        try:
            from ..core import autostart
            self.autostart_switch.setChecked(autostart.is_enabled())
        except Exception:
            self.autostart_switch.setEnabled(False)

        self.auto_bubble_switch.setChecked(
            cfg.get("ui", "auto_bubble", default=True))
        self.grammar_switch.setChecked(
            cfg.get("grammar", "enabled", default=False))
        self.subtitle_opacity_spin.setValue(
            cfg.get("subtitle", "opacity", default=100))
        self.input_opacity_spin.setValue(
            cfg.get("float_input", "opacity", default=100))
        self._update_color_button(
            self.subtitle_bg_button,
            cfg.get("subtitle", "bg_color", default="#121212"))
        self._update_color_button(
            self.subtitle_color_button,
            cfg.get("subtitle", "font_color", default="#ffffff"))
        weight = cfg.get("subtitle", "font_weight", default="bold")
        for i, (_, key) in enumerate(SUBTITLE_WEIGHTS):
            if key == weight:
                self.subtitle_weight_combo.setCurrentIndex(i)
                break
        self.subtitle_family_combo.setCurrentIndex(0)
        family = cfg.get("subtitle", "font_family", default="")
        if family:
            for i in range(1, self.subtitle_family_combo.count()):
                if self.subtitle_family_combo.itemText(i) == family:
                    self.subtitle_family_combo.setCurrentIndex(i)
                    break

        theme = cfg.get("ui", "theme", default="auto")
        for i, (_, value) in enumerate(THEMES):
            if value == theme:
                self.theme_combo.setCurrentIndex(i)
        color = cfg.get("ui", "theme_color", default="#0078d4")
        for i, (_, value) in enumerate(THEME_COLORS):
            if value.lower() == color.lower():
                self.color_combo.setCurrentIndex(i)

        sc = "system_captions"
        self.system_captions_switch.setChecked(
            cfg.get(sc, "enabled", default=False))
        self.system_device_combo.setCurrentIndex(0)
        device = cfg.get(sc, "device", default="default")
        if device != "default":
            for i in range(1, self.system_device_combo.count()):
                if self.system_device_combo.itemText(i) == device:
                    self.system_device_combo.setCurrentIndex(i)
                    break
        self.system_language_combo.setCurrentIndex(0)
        spoken = cfg.get(sc, "language", default="")
        if spoken:
            codes = [c for c, _ in LANGUAGES]
            if spoken in codes:
                self.system_language_combo.setCurrentIndex(
                    codes.index(spoken) + 1)
        engine = cfg.get(sc, "engine", default="nllb-600m")
        engine_ids = [e for e, _ in ENGINE_LABELS]
        if engine in engine_ids:
            self.system_engine_combo.setCurrentIndex(engine_ids.index(engine))
        self.system_compute_combo.setCurrentIndex(
            1 if cfg.get(sc, "compute_device", default="auto") == "cpu" else 0)
        self.system_rows_spin.setValue(cfg.get(sc, "display_rows", default=3))
        self.system_font_spin.setValue(cfg.get(sc, "font_size", default=20))
        self.system_opacity_spin.setValue(cfg.get(sc, "opacity", default=100))

    def _refresh_hotkey_button(self):
        self.hotkey_button.setText(hotkey_display(
            self.config.get("hotkey", "type", default="keyboard"),
            self.config.get("hotkey", "key", default="f9")))
        replay_key = self.config.get("replay_hotkey", "key", default="")
        self.replay_hotkey_button.setText(
            hotkey_display(
                self.config.get("replay_hotkey", "type", default="keyboard"),
                replay_key)
            if replay_key else "未設定")
        self.grammar_hotkey_button.setText(hotkey_display(
            self.config.get("grammar", "hotkey_type", default="keyboard"),
            self.config.get("grammar", "hotkey_key", default="f10")))
        self.system_hotkey_button.setText(hotkey_display(
            self.config.get("system_captions", "hotkey_type",
                            default="keyboard"),
            self.config.get("system_captions", "hotkey_key", default="f11")))

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

        self.source_lang_combo.currentIndexChanged.connect(
            lambda i: self._on_language_changed("source", i))
        self.target_lang_combo.currentIndexChanged.connect(
            lambda i: self._on_language_changed("target", i))
        self.stt_combo.currentIndexChanged.connect(self._on_stt_changed)
        self.hotkey_button.clicked.connect(self._on_hotkey_button)
        self.replay_hotkey_button.clicked.connect(self._on_replay_hotkey_button)
        self.replay_hotkey_clear.clicked.connect(self._on_replay_hotkey_clear)
        self.grammar_switch.checkedChanged.connect(self._on_grammar_switch)
        self.grammar_hotkey_button.clicked.connect(self._on_grammar_hotkey_button)

        self.auto_copy_switch.checkedChanged.connect(
            lambda checked: self._save("output", "auto_copy", checked))
        self.tts_switch.checkedChanged.connect(
            lambda checked: self._save("output", "tts_enabled", checked))
        self.tts_device_combo.currentIndexChanged.connect(
            lambda _: self._save("output", "tts_device",
                                 self.tts_device_combo.currentText()))
        self.tts_rate_spin.valueChanged.connect(
            lambda v: self._save("output", "tts_rate", v))
        self.tts_volume_spin.valueChanged.connect(
            lambda v: self._save("output", "tts_volume", v))
        self.mic_device_combo.currentIndexChanged.connect(self._on_mic_device_changed)
        self.isolate_switch.checkedChanged.connect(
            lambda checked: self._save(
                "recording", "isolate_other_devices", checked))
        self.subtitle_duration_spin.valueChanged.connect(
            lambda v: self._save("subtitle", "duration_seconds", v))
        self.subtitle_font_spin.valueChanged.connect(
            lambda v: self._save("subtitle", "font_size", v))
        self.float_input_switch.checkedChanged.connect(self._on_float_input)
        self.subtitle_opacity_spin.valueChanged.connect(
            self._on_opacity_changed)
        self.input_opacity_spin.valueChanged.connect(self._on_opacity_changed)
        self.subtitle_bg_button.clicked.connect(
            lambda: self._pick_subtitle_color(
                "bg_color", "選擇字幕底色", self.subtitle_bg_button))
        self.subtitle_color_button.clicked.connect(
            lambda: self._pick_subtitle_color(
                "font_color", "選擇字體顏色", self.subtitle_color_button))
        self.subtitle_weight_combo.currentIndexChanged.connect(
            self._on_subtitle_weight)
        self.subtitle_family_combo.currentIndexChanged.connect(
            self._on_subtitle_family)
        self.autostart_switch.checkedChanged.connect(self._on_autostart)
        self.auto_bubble_switch.checkedChanged.connect(
            lambda checked: self._save("ui", "auto_bubble", checked))

        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self.color_combo.currentIndexChanged.connect(self._on_color_changed)

        self.system_captions_switch.checkedChanged.connect(
            self._on_system_captions_switch)
        self.system_hotkey_button.clicked.connect(
            self._on_system_hotkey_button)
        self.system_device_combo.currentIndexChanged.connect(
            self._on_system_settings_changed)
        self.system_language_combo.currentIndexChanged.connect(
            self._on_system_settings_changed)
        self.system_engine_combo.currentIndexChanged.connect(
            self._on_system_settings_changed)
        self.system_compute_combo.currentIndexChanged.connect(
            self._on_system_settings_changed)
        self.system_rows_spin.valueChanged.connect(
            self._on_system_settings_changed)
        self.system_font_spin.valueChanged.connect(
            self._on_system_settings_changed)
        self.system_opacity_spin.valueChanged.connect(
            self._on_system_settings_changed)

    def _save(self, *keys_and_value):
        if self._loading:
            return
        self.config.set(*keys_and_value)

    def _on_mic_device_changed(self, index: int):
        if self._loading or index < 0:
            return
        device = (DEFAULT_DEVICE if index == 0
                  else self.mic_device_combo.currentText())
        self.config.set("recording", "device", device)
        self.mic_device_changed.emit()

    def _on_stt_changed(self, index: int):
        if self._loading:
            return
        model = STT_MODELS[index]
        self.config.set("stt", "model_size", model)
        self.stt_model_changed.emit(model)

    @staticmethod
    def _update_color_button(button, color_hex: str):
        button.setText(color_hex)
        text_color = "black" if QColor(color_hex).lightness() > 128 else "white"
        button.setStyleSheet(
            f"PushButton {{ background-color: {color_hex};"
            f" color: {text_color}; }}")

    def _pick_subtitle_color(self, key: str, title: str, button):
        current = QColor(self.config.get(
            "subtitle", key,
            default="#121212" if key == "bg_color" else "#ffffff"))
        dialog = ColorDialog(current, title, self.window(), enableAlpha=False)
        dialog.colorChanged.connect(
            lambda c: self._apply_subtitle_color(key, c, button))
        dialog.exec()

    def _apply_subtitle_color(self, key: str, color: QColor, button):
        self.config.set("subtitle", key, color.name())
        self._update_color_button(button, color.name())
        self.subtitle_style_changed.emit()

    def _on_subtitle_weight(self, index: int):
        if self._loading or index < 0:
            return
        self.config.set("subtitle", "font_weight", SUBTITLE_WEIGHTS[index][1])
        self.subtitle_style_changed.emit()

    def _on_subtitle_family(self, index: int):
        if self._loading or index < 0:
            return
        family = "" if index == 0 else self.subtitle_family_combo.currentText()
        self.config.set("subtitle", "font_family", family)
        self.subtitle_style_changed.emit()

    def _on_opacity_changed(self, _value: int):
        if self._loading:
            return
        self.config.set("subtitle", "opacity",
                        self.subtitle_opacity_spin.value())
        self.config.set("float_input", "opacity",
                        self.input_opacity_spin.value())
        self.overlay_opacity_changed.emit()

    def _on_float_input(self, checked: bool):
        if self._loading:
            return
        # config 寫入由 MainWindow.set_float_input_enabled 統一處理
        self.float_input_toggled.emit(checked)

    def set_tts_rate(self, value: int):
        """由字幕上的語速拖桿回寫，同步設定頁的顯示。"""
        self._loading = True
        self.tts_rate_spin.setValue(int(value))
        self._loading = False

    def set_tts_volume(self, value: int):
        """由字幕上的音量拖桿回寫，同步設定頁的顯示。"""
        self._loading = True
        self.tts_volume_spin.setValue(int(value))
        self._loading = False

    def set_float_input_checked(self, checked: bool):
        """由主視窗回寫（懸浮球選單/輸入框 ✕ 改了狀態時同步 UI）。"""
        self._loading = True
        self.float_input_switch.setChecked(checked)
        self._loading = False

    def _on_autostart(self, checked: bool):
        if self._loading:
            return
        try:
            from ..core import autostart
            autostart.set_enabled(checked)
        except Exception as e:
            self.error_requested.emit(f"設定開機自動啟動失敗：{e}")
            self._loading = True
            self.autostart_switch.setChecked(not checked)
            self._loading = False

    def _on_language_changed(self, which: str, index: int):
        if self._loading or index < 0:
            return
        code = LANGUAGES[index][0]
        other = self.config.get(
            "language", "target" if which == "source" else "source",
            default="en" if which == "source" else "zh")
        if code == other:
            self.error_requested.emit("母語和目標語言不能相同")
            self._loading = True
            self._load_from_config()
            self._loading = False
            return
        self.config.set("language", which, code)
        self.languages_changed.emit()

    def _on_reset_defaults(self):
        from qfluentwidgets import MessageBox
        box = MessageBox(
            "全部恢復預設值",
            "確定要把所有設定恢復成預設值嗎？\nDeepSeek API Key 會保留，其他設定全部重置。",
            self.window())
        if not box.exec():
            return
        self.config.reset_to_defaults()
        self._loading = True
        self._load_from_config()
        self._loading = False
        self.settings_reset.emit()

    def _on_hotkey_button(self):
        # 擷取期間先停掉現用的熱鍵監聽，避免按到現用熱鍵誤觸錄音
        self.hotkey_capture_started.emit()
        dialog = HotkeyCaptureDialog(self.window())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            taken = self._existing_hotkeys(exclude="hotkey")
            if (dialog.result_type, dialog.result_key) in taken:
                self.error_requested.emit("錄音熱鍵不能跟其他熱鍵相同")
            else:
                self.config.set("hotkey", "type", dialog.result_type)
                self.config.set("hotkey", "key", dialog.result_key)
                self._refresh_hotkey_button()
        # 取消也要 emit：讓主視窗重新掛回熱鍵監聽
        self.hotkey_changed.emit()

    def _on_replay_hotkey_button(self):
        self.hotkey_capture_started.emit()
        dialog = HotkeyCaptureDialog(self.window())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            taken = self._existing_hotkeys(exclude="replay_hotkey")
            if (dialog.result_type, dialog.result_key) in taken:
                self.error_requested.emit("朗讀熱鍵不能跟其他熱鍵相同")
            else:
                self.config.set("replay_hotkey", "type", dialog.result_type)
                self.config.set("replay_hotkey", "key", dialog.result_key)
                self._refresh_hotkey_button()
        self.hotkey_changed.emit()

    def _on_replay_hotkey_clear(self):
        self.config.set("replay_hotkey", "key", "")
        self._refresh_hotkey_button()
        self.hotkey_changed.emit()

    def _on_grammar_switch(self, checked: bool):
        if self._loading:
            return
        self.config.set("grammar", "enabled", checked)
        self.hotkey_changed.emit()  # 讓主視窗重新套用熱鍵監聽

    # 各熱鍵在 config 裡的 (區段, 型別鍵, 按鍵鍵, 預設按鍵)
    _HOTKEY_SLOTS = (
        ("hotkey", "type", "key", "f9"),
        ("replay_hotkey", "type", "key", ""),
        ("grammar", "hotkey_type", "hotkey_key", "f10"),
        ("system_captions", "hotkey_type", "hotkey_key", "f11"),
    )

    def _existing_hotkeys(self, exclude=None):
        """目前已被占用的熱鍵。exclude 是要排除的區段名——
        重新設定同一個熱鍵時，不該跟自己現在的設定判定為衝突。"""
        taken = set()
        for section, type_key, key_key, default in self._HOTKEY_SLOTS:
            if section == exclude:
                continue
            key = self.config.get(section, key_key, default=default)
            if not key:
                continue
            taken.add(
                (self.config.get(section, type_key, default="keyboard"), key))
        return taken

    def _on_grammar_hotkey_button(self):
        self.hotkey_capture_started.emit()
        dialog = HotkeyCaptureDialog(self.window())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            taken = self._existing_hotkeys(exclude="grammar")
            if (dialog.result_type, dialog.result_key) in taken:
                self.error_requested.emit("文法檢查熱鍵不能跟其他熱鍵相同")
            else:
                self.config.set("grammar", "hotkey_type", dialog.result_type)
                self.config.set("grammar", "hotkey_key", dialog.result_key)
                self._refresh_hotkey_button()
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

    def _on_system_captions_switch(self, checked: bool):
        if self._loading:
            return
        # config 寫入由 MainWindow.set_system_captions_enabled 統一處理
        self.system_captions_toggled.emit(checked)

    def set_system_captions_checked(self, checked: bool):
        """由主視窗回寫（熱鍵或字幕 ✕ 改了狀態時同步 UI）。"""
        self._loading = True
        self.system_captions_switch.setChecked(checked)
        self._loading = False

    def _on_system_settings_changed(self, _value=None):
        if self._loading:
            return
        sc = "system_captions"
        pipeline_keys = ("device", "engine", "compute_device")
        before = tuple(self.config.get(sc, k) for k in pipeline_keys)
        index = self.system_device_combo.currentIndex()
        self.config.set(sc, "device", "default" if index == 0
                        else self.system_device_combo.currentText())
        lang_index = self.system_language_combo.currentIndex()
        self.config.set(sc, "language", "" if lang_index == 0
                        else LANGUAGES[lang_index - 1][0])
        self.config.set(sc, "engine",
                        ENGINE_LABELS[self.system_engine_combo.currentIndex()][0])
        self.config.set(sc, "compute_device",
                        "cpu" if self.system_compute_combo.currentIndex() == 1
                        else "auto")
        self.config.set(sc, "display_rows", self.system_rows_spin.value())
        self.config.set(sc, "font_size", self.system_font_spin.value())
        self.config.set(sc, "opacity", self.system_opacity_spin.value())
        self.system_captions_settings_changed.emit()
        after = tuple(self.config.get(sc, k) for k in pipeline_keys)
        if after != before:
            # 這幾項只在 start() 時讀取，正在跑的話得重啟才會生效
            self.system_captions_pipeline_changed.emit()

    def _on_system_hotkey_button(self):
        self.hotkey_capture_started.emit()
        dialog = HotkeyCaptureDialog(self.window())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            taken = self._existing_hotkeys(exclude="system_captions")
            if (dialog.result_type, dialog.result_key) in taken:
                self.error_requested.emit("系統字幕熱鍵不能跟其他熱鍵相同")
            else:
                self.config.set("system_captions", "hotkey_type",
                                dialog.result_type)
                self.config.set("system_captions", "hotkey_key",
                                dialog.result_key)
                self._refresh_hotkey_button()
        self.hotkey_changed.emit()
