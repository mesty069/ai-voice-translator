from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget

from qfluentwidgets import (
    CaptionLabel,
    FluentIcon,
    FluentWindow,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
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
from .settings_page import SettingsInterface

STATE_COLORS = {
    "idle": "#2ecc71",
    "loading": "#f39c12",
    "recording": "#e74c3c",
    "processing": "#3498db",
    "error": "#e74c3c",
}

HOTKEY_DISPLAY = {"keyboard": "鍵盤", "mouse": "滑鼠"}


class HomeInterface(QWidget):
    translate_requested = Signal(str)

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
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        self.raw_edit = self._add_section(layout, "原始辨識中文")
        self.refined_edit = self._add_section(layout, "梳理後中文")
        self.english_edit = self._add_section(layout, "英文翻譯")

        self.copy_button = PrimaryPushButton(FluentIcon.COPY, "複製英文", self)
        self.copy_button.clicked.connect(
            lambda: self.copy_text(self.english_edit.toPlainText()))
        layout.addWidget(self.copy_button)

        layout.addSpacing(4)
        layout.addWidget(CaptionLabel("或直接輸入中文（Ctrl+Enter 送出）", self))
        self.input_edit = TextEdit(self)
        self.input_edit.setFixedHeight(64)
        self.input_edit.setPlaceholderText("在這裡輸入中文，AI 一樣會先梳理再翻譯")
        layout.addWidget(self.input_edit)
        self.translate_button = PushButton(FluentIcon.SEND, "翻譯輸入的文字", self)
        self.translate_button.clicked.connect(self._submit_text)
        layout.addWidget(self.translate_button)
        QShortcut(QKeySequence("Ctrl+Return"), self.input_edit,
                  self._submit_text,
                  context=Qt.ShortcutContext.WidgetShortcut)

        self.refresh_hint()
        self.set_state("loading", "啟動中…")

    def _add_section(self, layout: QVBoxLayout, title: str) -> TextEdit:
        row = QHBoxLayout()
        row.addWidget(CaptionLabel(title, self))
        row.addStretch(1)
        edit = TextEdit(self)
        edit.setReadOnly(True)
        edit.setFixedHeight(88)
        copy_btn = ToolButton(FluentIcon.COPY, self)
        copy_btn.setFixedSize(28, 28)
        copy_btn.clicked.connect(lambda: self.copy_text(edit.toPlainText()))
        row.addWidget(copy_btn)
        layout.addLayout(row)
        layout.addWidget(edit)
        return edit

    def refresh_hint(self):
        hotkey_type = self.config.get("hotkey", "type", default="keyboard")
        key = self.config.get("hotkey", "key", default="f9").upper()
        type_name = HOTKEY_DISPLAY.get(hotkey_type, hotkey_type)
        self.hint_label.setText(f"按住 {type_name} {key} 說中文，放開後自動翻譯成英文")

    def set_state(self, state: str, message: str):
        color = STATE_COLORS.get(state, "#95a5a6")
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 14px;")
        self.status_label.setText(message)

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
        self._force_quit = False

        self.setWindowTitle("AI 語音中翻英")
        self.resize(760, 760)

        self.home = HomeInterface(config, self)
        self.settings = SettingsInterface(config, self)
        self.addSubInterface(self.home, FluentIcon.MICROPHONE, "翻譯")
        self.addSubInterface(self.settings, FluentIcon.SETTING, "設定")

        self.bubble = BubbleWidget()
        self.bubble.clicked.connect(self._restore_from_bubble)
        self.bubble.close_requested.connect(self._quit)

        controller.state_changed.connect(self.home.set_state)
        controller.state_changed.connect(
            lambda state, _msg: self.bubble.set_state(state))
        controller.result_ready.connect(self.home.show_result)
        controller.error_occurred.connect(self._show_error)
        controller.copy_requested.connect(
            lambda text: QApplication.clipboard().setText(text))

        self.home.bubble_button.clicked.connect(self.hide_to_bubble)
        self.home.translate_requested.connect(controller.translate_text)
        self.settings.hotkey_changed.connect(self._on_hotkey_changed)
        self.settings.stt_model_changed.connect(controller.reload_model)

    def _on_hotkey_changed(self):
        self.controller.apply_hotkey()
        self.home.refresh_hint()

    def _show_error(self, message: str):
        InfoBar.error(
            "錯誤", message, parent=self, duration=5000,
            position=InfoBarPosition.TOP_RIGHT)

    def hide_to_bubble(self):
        self.hide()
        self.bubble.show()

    def _restore_from_bubble(self):
        self.bubble.hide()
        self.showNormal()
        self.activateWindow()

    def _quit(self):
        self._force_quit = True
        self.bubble.hide()
        self.controller.shutdown()
        QApplication.quit()

    def closeEvent(self, event):
        # Qt6 的 quit() 會先對所有視窗發 close 事件，被 ignore 會取消 quit，
        # 所以真正退出時必須 accept；平常按 X 則收成懸浮球
        if self._force_quit:
            event.accept()
            return
        event.ignore()
        self.hide_to_bubble()
