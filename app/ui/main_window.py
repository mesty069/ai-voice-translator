import math

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget

from qfluentwidgets import (
    CaptionLabel,
    FluentIcon,
    FluentWindow,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    StrongBodyLabel,
    TextEdit,
    TitleLabel,
    ToolButton,
    TransparentToolButton,
)

from ..config import Config
from ..controller import AppController
from .bubble import BubbleWidget
from .float_input import EnterSubmitFilter, FloatingInputWidget
from .settings_page import SettingsInterface
from .subtitle import SubtitleOverlay
from .wait_hint import WaitHintOverlay

STATE_COLORS = {
    "idle": "#2ecc71",
    "loading": "#f39c12",
    "recording": "#e74c3c",
    "processing": "#3498db",
    "error": "#e74c3c",
}

class HomeInterface(QWidget):
    translate_requested = Signal(str)
    speak_requested = Signal(str)

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.setObjectName("homeInterface")
        self.config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        self.hint_label = TitleLabel("", self)
        top_row.addWidget(self.hint_label)
        top_row.addStretch(1)
        self.bubble_button = TransparentToolButton(FluentIcon.MINIMIZE, self)
        self.bubble_button.setToolTip("縮成懸浮球")
        top_row.addWidget(self.bubble_button)
        layout.addLayout(top_row)

        status_row = QHBoxLayout()
        self.status_dot = CaptionLabel("●", self)
        self.status_label = StrongBodyLabel("啟動中…", self)
        self.status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label)
        status_row.addStretch(1)
        # 錄音時的即時峰值表（-60dB~0dB 對映 0~100）
        self.level_label = CaptionLabel("", self)
        self.level_bar = ProgressBar(self)
        self.level_bar.setRange(0, 100)
        self.level_bar.setFixedWidth(140)
        self.level_bar.setTextVisible(False)
        status_row.addWidget(self.level_label)
        status_row.addWidget(self.level_bar)
        self.level_label.hide()
        self.level_bar.hide()
        layout.addLayout(status_row)

        self.raw_edit = self._add_section(layout, "原始辨識")
        self.refined_edit = self._add_section(layout, "梳理後原文")
        self.english_edit = self._add_section(layout, "翻譯結果", tts=True)

        self.copy_button = PrimaryPushButton(FluentIcon.COPY, "複製翻譯", self)
        self.copy_button.clicked.connect(
            lambda: self.copy_text(self.english_edit.toPlainText()))
        layout.addWidget(self.copy_button)

        layout.addSpacing(4)
        layout.addWidget(CaptionLabel("或直接輸入文字（Enter 送出，Ctrl+Enter 換行）", self))
        self.input_edit = TextEdit(self)
        self.input_edit.setFixedHeight(64)
        self.input_edit.setPlaceholderText("在這裡輸入母語文字，AI 一樣會先梳理再翻譯")
        layout.addWidget(self.input_edit)
        self.translate_button = PushButton(FluentIcon.SEND, "翻譯輸入的文字", self)
        self.translate_button.clicked.connect(self._submit_text)
        layout.addWidget(self.translate_button)
        EnterSubmitFilter(self.input_edit, self._submit_text)

        self.refresh_hint()
        self.set_state("loading", "啟動中…")

    def _add_section(self, layout: QVBoxLayout, title: str,
                     tts: bool = False) -> TextEdit:
        row = QHBoxLayout()
        row.addWidget(CaptionLabel(title, self))
        row.addStretch(1)
        edit = TextEdit(self)
        edit.setReadOnly(True)
        edit.setFixedHeight(88)
        if tts:
            tts_btn = ToolButton(FluentIcon.VOLUME, self)
            tts_btn.setFixedSize(28, 28)
            tts_btn.setToolTip("朗讀英文")
            tts_btn.clicked.connect(
                lambda: self.speak_requested.emit(edit.toPlainText()))
            row.addWidget(tts_btn)
        copy_btn = ToolButton(FluentIcon.COPY, self)
        copy_btn.setFixedSize(28, 28)
        copy_btn.clicked.connect(lambda: self.copy_text(edit.toPlainText()))
        row.addWidget(copy_btn)
        layout.addLayout(row)
        layout.addWidget(edit)
        return edit

    def refresh_hint(self):
        from ..config import lang_display
        from .settings_page import hotkey_display
        name = hotkey_display(
            self.config.get("hotkey", "type", default="keyboard"),
            self.config.get("hotkey", "key", default="f9"))
        src = lang_display(self.config.get("language", "source", default="zh"))
        tgt = lang_display(self.config.get("language", "target", default="en"))
        self.hint_label.setText(
            f"按住 {name} 說{src}，放開後自動翻譯成{tgt}")

    def set_state(self, state: str, message: str):
        color = STATE_COLORS.get(state, "#95a5a6")
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 14px;")
        self.status_label.setText(message)
        recording = state == "recording"
        self.level_label.setVisible(recording)
        self.level_bar.setVisible(recording)
        if not recording:
            self.level_bar.setValue(0)
            self.level_label.setText("")

    def update_level(self, peak: float):
        db = 20 * math.log10(max(peak, 1e-6))
        self.level_bar.setValue(int(max(0.0, min(1.0, (db + 60) / 60)) * 100))
        self.level_label.setText(f"峰值 {peak:.2f}")

    def show_result(self, raw: str, refined: str, english: str):
        self.raw_edit.setPlainText(raw)
        self.refined_edit.setPlainText(refined)
        self.english_edit.setPlainText(english)

    def copy_text(self, text: str):
        if not text:
            return
        QApplication.clipboard().setText(text)
        InfoBar.success(
            "已複製", "", parent=self, duration=1500,
            position=InfoBarPosition.TOP_RIGHT)

    def _submit_text(self):
        text = self.input_edit.toPlainText().strip()
        if text:
            self.translate_requested.emit(text)


class MainWindow(FluentWindow):
    def __init__(self, config: Config, controller: AppController):
        super().__init__()
        self.config = config
        self.controller = controller
        self._quitting = False
        self._open_error_bars = {}  # message -> InfoBar（防止同樣錯誤疊加）
        self._last_result = None    # (refined, english) 最後一筆成功結果
        self._app_state = QApplication.applicationState  # 可注入以便測試
        QApplication.instance().applicationStateChanged.connect(
            self._on_app_state_changed)
        self._saved_geometry = None
        self._transitioning = False  # 收縮/還原動畫進行中，防止重入
        self._transition_done_cb = None

        # 動畫物件只建一次重複使用：在 finished 回呼中重建/丟棄動畫物件
        # 會讓 PySide 在訊號發送途中刪除 C++ 物件而崩潰
        self._geo_anim = QPropertyAnimation(self, b"geometry", self)
        self._geo_anim.setDuration(200)
        self._opacity_anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._opacity_anim.setDuration(200)
        self._anim_group = QParallelAnimationGroup(self)
        self._anim_group.addAnimation(self._geo_anim)
        self._anim_group.addAnimation(self._opacity_anim)
        self._anim_group.finished.connect(self._on_transition_done)

        self.setWindowTitle("AI 語音中翻英")
        self.resize(760, 760)

        self.home = HomeInterface(config, self)
        self.settings = SettingsInterface(config, self)
        self.addSubInterface(self.home, FluentIcon.MICROPHONE, "翻譯")
        self.addSubInterface(self.settings, FluentIcon.SETTING, "設定")

        self.bubble = BubbleWidget()
        self.bubble.clicked.connect(self._restore_from_bubble)
        self.bubble.close_requested.connect(self._quit)
        self.subtitle = SubtitleOverlay(config)
        self.subtitle.replay_requested.connect(controller.replay_tts)
        self.subtitle.rate_changed.connect(self.settings.set_tts_rate)
        self.subtitle.retranslate_requested.connect(controller.translate_text)
        self.wait_hint = WaitHintOverlay()
        controller.wait_hint_visible.connect(self._on_wait_hint)
        self.float_input = FloatingInputWidget(config)
        self.float_input.translate_requested.connect(controller.translate_text)
        self.float_input.closed_by_user.connect(
            lambda: self.set_float_input_enabled(False))
        self.bubble.input_toggle_requested.connect(self._toggle_float_input)

        controller.state_changed.connect(self.home.set_state)
        controller.state_changed.connect(
            lambda state, _msg: self.bubble.set_state(state))
        controller.state_changed.connect(self._on_state_changed)
        controller.result_ready.connect(self.home.show_result)
        controller.result_ready.connect(self._on_result_ready)
        controller.level_changed.connect(self.home.update_level)
        controller.error_occurred.connect(self._show_error)
        controller.copy_requested.connect(
            lambda text: QApplication.clipboard().setText(text))

        self.home.bubble_button.clicked.connect(self.hide_to_bubble)
        self.home.translate_requested.connect(controller.translate_text)
        self.home.speak_requested.connect(controller.speak_text)
        self.settings.hotkey_changed.connect(self._on_hotkey_changed)
        self.settings.hotkey_capture_started.connect(controller.hotkey.stop)
        self.settings.hotkey_capture_started.connect(controller.replay_hotkey.stop)
        self.settings.hotkey_capture_started.connect(controller.grammar_hotkey.stop)
        self.settings.error_requested.connect(self._show_error)
        self.settings.settings_reset.connect(self._on_settings_reset)
        self.settings.languages_changed.connect(self.home.refresh_hint)
        controller.replay_hotkey_pressed.connect(self._on_replay_hotkey)
        controller.enter_pressed.connect(self._on_global_enter)
        controller.tts_playing.connect(self.subtitle.set_reading)
        self.settings.stt_model_changed.connect(controller.reload_model)
        self.settings.mic_device_changed.connect(controller.apply_recording_device)
        self.settings.float_input_toggled.connect(self.set_float_input_enabled)
        self.settings.overlay_opacity_changed.connect(self._apply_overlay_opacity)
        self.settings.subtitle_style_changed.connect(self.subtitle.refresh_style)

    def _on_hotkey_changed(self):
        self.controller.apply_hotkey()
        self.controller.apply_replay_hotkey()
        self.controller.apply_grammar_hotkey()
        self.home.refresh_hint()

    def _on_app_state_changed(self, state):
        """使用者切到別的程式（整個 app 失去前景）→ 自動縮成懸浮球。"""
        if state != Qt.ApplicationState.ApplicationInactive:
            return
        if not self.config.get("ui", "auto_bubble", default=True):
            return
        if not self.isVisible() or self._quitting or self._transitioning:
            return
        # 短暫的焦點抖動不算，200ms 後仍在背景才收
        QTimer.singleShot(200, self._auto_bubble_check)

    def _auto_bubble_check(self):
        if (self._app_state() == Qt.ApplicationState.ApplicationInactive
                and self.isVisible() and not self._quitting
                and not self._transitioning):
            self.hide_to_bubble()

    def _on_settings_reset(self):
        """恢復預設後把所有會即時生效的部分重新套用。"""
        c = self.controller
        c.apply_hotkey()
        c.apply_replay_hotkey()
        c.apply_grammar_hotkey()
        c.apply_recording_device()
        self.home.refresh_hint()
        self.set_float_input_enabled(
            self.config.get("float_input", "enabled", default=False))
        self.subtitle.refresh_style()
        self._apply_overlay_opacity()
        model = self.config.get("stt", "model_size", default="large-v3")
        if model != c.stt.model_size:
            c.reload_model(model)

    def _on_wait_hint(self, visible: bool):
        # 顯示在使用者目前所在的螢幕（視窗開著看視窗，否則看懸浮球）
        screen = self.screen() if self.isVisible() else self.bubble.screen()
        self.wait_hint.set_visible(visible, screen)

    def _on_replay_hotkey(self):
        if self.subtitle.isVisible():
            self.controller.replay_tts()
            return
        # 字幕已消失：帶上一筆結果重新顯示字幕並播放
        if self._last_result is None:
            return
        if not self.isVisible():  # 主視窗開著時結果就在畫面上，只重播聲音
            refined, english = self._last_result
            self.subtitle.show_result(refined, english, self.bubble.screen())
        self.controller.replay_tts()

    def _on_global_enter(self):
        """字幕顯示中直接按 Enter → 進入輸入模式（打新句子）。"""
        if not self.subtitle.isVisible() or self.subtitle._editing:
            return
        # 剛用 Enter 送出編輯時，全域監聽也會收到同一個 Enter → 冷卻期忽略
        import time
        if time.monotonic() - self.subtitle.last_edit_finished < 0.5:
            return
        # 焦點在本程式「可見的文字輸入框」（浮動輸入框、主視窗輸入區）時，
        # Enter 屬於那個輸入框，不攔；按鈕或隱藏視窗殘留的焦點不算
        from PySide6.QtWidgets import QLineEdit, QPlainTextEdit, QTextEdit
        focus = QApplication.focusWidget()
        if (focus is not None and focus.isVisible()
                and isinstance(focus, (QTextEdit, QPlainTextEdit, QLineEdit))):
            return
        self.subtitle.begin_edit_empty()

    def _apply_overlay_opacity(self):
        self.subtitle.apply_opacity()
        self.float_input.apply_opacity()

    def _toggle_float_input(self):
        self.set_float_input_enabled(
            not self.config.get("float_input", "enabled", default=False))

    def set_float_input_enabled(self, enabled: bool):
        """懸浮球選單、設定頁開關、輸入框 ✕ 三處共用的同一個狀態。"""
        self.config.set("float_input", "enabled", enabled)
        self.settings.set_float_input_checked(enabled)
        if enabled and not self.isVisible():
            self.float_input.show_overlay(self.bubble.screen())
        else:
            self.float_input.hide()

    def _on_state_changed(self, state: str, message: str):
        # 翻譯流程結束（成功或失敗）→ 收掉等待旋轉圈
        if state in ("idle", "error"):
            self.subtitle.set_busy(False)
            self.float_input.set_busy(False)
        # 錯誤：視窗開著→可複製的錯誤條；懸浮球模式→字幕顯示
        if state == "error":
            self._show_error(message)

    def _on_result_ready(self, _raw: str, refined: str, english: str):
        if english:  # 記住最後一筆成功結果，朗讀熱鍵可叫回字幕
            self._last_result = (refined, english)
        # 懸浮球模式下用字幕顯示結果
        if not self.isVisible():
            self.subtitle.show_result(refined, english, self.bubble.screen())

    def _show_error(self, message: str):
        # 懸浮球模式 → 用字幕顯示
        if not self.isVisible():
            self.subtitle.show_message(f"⚠ {message}", self.bubble.screen())
            return
        # 同樣的錯誤條還開著就不要重複疊加
        existing = self._open_error_bars.get(message)
        if existing is not None:
            try:
                if existing.isVisible():
                    return
            except RuntimeError:
                pass  # C++ 物件已被刪除
            self._open_error_bars.pop(message, None)
        # duration=-1：錯誤不自動消失，方便閱讀與複製，按 X 關閉
        bar = InfoBar.error(
            "錯誤", message, parent=self, duration=-1,
            position=InfoBarPosition.TOP_RIGHT)
        copy_btn = PushButton("複製", bar)
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(message))
        bar.addWidget(copy_btn)
        self._open_error_bars[message] = bar

    def _geometry_toward_bubble(self, geo: QRect) -> QRect:
        """視窗縮小並朝懸浮球方向偏移後的幾何，作為收縮/還原動畫的端點。"""
        bubble_center = self.bubble.frameGeometry().center()
        w = int(geo.width() * 0.72)
        h = int(geo.height() * 0.72)
        cx = geo.center().x() + int((bubble_center.x() - geo.center().x()) * 0.35)
        cy = geo.center().y() + int((bubble_center.y() - geo.center().y()) * 0.35)
        return QRect(cx - w // 2, cy - h // 2, w, h)

    def _on_transition_done(self):
        self._transitioning = False
        cb, self._transition_done_cb = self._transition_done_cb, None
        if cb is not None:
            cb()

    def _animate_window(self, start: QRect, end: QRect,
                        fade_in: bool, easing, on_done):
        self._transition_done_cb = None
        self._anim_group.stop()
        self._transitioning = True
        self._transition_done_cb = on_done
        self._geo_anim.setEasingCurve(easing)
        self._geo_anim.setStartValue(start)
        self._geo_anim.setEndValue(end)
        self._opacity_anim.setStartValue(0.0 if fade_in else 1.0)
        self._opacity_anim.setEndValue(1.0 if fade_in else 0.0)
        self._anim_group.start()

    def hide_to_bubble(self):
        if self._transitioning or not self.isVisible():
            return
        self._saved_geometry = self.geometry()

        def _done():
            self.hide()
            self.setGeometry(self._saved_geometry)
            self.setWindowOpacity(1.0)
            self.bubble.show_animated()
            if self.config.get("float_input", "enabled", default=False):
                self.float_input.show_overlay(self.bubble.screen())

        self._animate_window(
            self.geometry(), self._geometry_toward_bubble(self._saved_geometry),
            fade_in=False, easing=QEasingCurve.Type.InCubic, on_done=_done)

    def _restore_from_bubble(self):
        if self._transitioning or self.isVisible():
            return
        self.bubble.hide_animated(self._show_from_bubble)

    def _show_from_bubble(self):
        self.subtitle.dismiss()
        self.float_input.hide()
        geo = self._saved_geometry or self.geometry()
        self._saved_geometry = geo
        self.setGeometry(self._geometry_toward_bubble(geo))
        self.setWindowOpacity(0.0)
        self.showNormal()
        self.activateWindow()

        def _done():
            self.setGeometry(geo)
            self.setWindowOpacity(1.0)

        self._animate_window(
            self.geometry(), geo,
            fade_in=True, easing=QEasingCurve.Type.OutCubic, on_done=_done)

    def _quit(self):
        if self._quitting:
            return
        self._quitting = True
        self.subtitle.hide()
        self.float_input.hide()
        self.wait_hint.hide()
        self.bubble.hide()
        self.controller.shutdown()
        QApplication.quit()

    def closeEvent(self, event):
        # 按 X＝真正結束程式（縮小才是收成懸浮球）
        event.accept()
        self._quit()

    def changeEvent(self, event):
        # 標題列的縮小鈕 → 收成懸浮球（不進工作列）
        if (event.type() == QEvent.Type.WindowStateChange
                and self.isMinimized() and not self._quitting):
            QTimer.singleShot(0, self._minimize_to_bubble)
        super().changeEvent(event)

    def _minimize_to_bubble(self):
        if self._quitting or not self.isMinimized():
            return
        self._saved_geometry = self.normalGeometry()
        self.setWindowState(Qt.WindowState.WindowNoState)
        self.hide()
        self.bubble.show_animated()
        if self.config.get("float_input", "enabled", default=False):
            self.float_input.show_overlay(self.bubble.screen())
