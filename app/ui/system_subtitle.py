from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
)

from qfluentwidgets import FluentIcon, Theme, TransparentToolButton

from .overlay_base import DraggableResizableOverlay

_FLAGS = (Qt.WindowType.FramelessWindowHint
          | Qt.WindowType.WindowStaysOnTopHint
          | Qt.WindowType.Tool
          | Qt.WindowType.WindowDoesNotAcceptFocus)

DEFAULT_BG = "#0d2b33"
ACCENT = "#4dd0e1"      # 青色，與麥克風字幕（深灰白字）明顯區分
MIN_SIZE = QSize(360, 130)


class SystemSubtitleOverlay(DraggableResizableOverlay):
    """系統聲音的雙語字幕：上行原文、下行母語翻譯。

    與麥克風字幕的差異：青藍配色、左上角「🔊 系統聲音」標籤、
    不會自動倒數消失（停止擷取才收起）、可展開本次逐字稿。
    """

    CONFIG_SECTION = "system_captions"
    closed_by_user = Signal()

    def __init__(self, config):
        super().__init__(config, _FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMinimumSize(MIN_SIZE)
        self._history = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 10, 12, 16)
        outer.setSpacing(6)

        header = QHBoxLayout()
        self.tag_label = QLabel("🔊 系統聲音", self)
        self.tag_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 12px; font-weight: bold;"
            "font-family: 'Microsoft JhengHei';")
        header.addWidget(self.tag_label)
        header.addStretch(1)
        self.history_button = TransparentToolButton(
            FluentIcon.HISTORY.icon(Theme.DARK), self)
        self.history_button.setToolTip("展開／收起這次的逐字稿")
        self.history_button.setFixedSize(22, 22)
        self.history_button.clicked.connect(self._toggle_history)
        header.addWidget(self.history_button)
        self.close_button = TransparentToolButton(
            FluentIcon.CLOSE.icon(Theme.DARK), self)
        self.close_button.setToolTip("關閉系統聲音字幕")
        self.close_button.setFixedSize(22, 22)
        self.close_button.clicked.connect(self._on_close)
        header.addWidget(self.close_button)
        outer.addLayout(header)

        self.original_label = QLabel("", self)
        self.original_label.setWordWrap(True)
        outer.addWidget(self.original_label)

        self.translated_label = QLabel("", self)
        self.translated_label.setWordWrap(True)
        outer.addWidget(self.translated_label)
        outer.addStretch(1)

        self.history_view = QTextEdit(self)
        self.history_view.setReadOnly(True)
        self.history_view.hide()
        self.history_view.setStyleSheet(
            "QTextEdit { background: rgba(255,255,255,18); color: white;"
            " border: 1px solid rgba(255,255,255,45); border-radius: 6px;"
            " font-size: 13px; font-family: 'Microsoft JhengHei'; }")
        outer.addWidget(self.history_view)

        self.apply_style()

    # ---- 對外 API ----

    def show_overlay(self, screen=None):
        pos = self.saved_pos()
        if pos is not None:
            screen = QApplication.screenAt(pos)
        screen = screen or QApplication.primaryScreen()
        geo = screen.availableGeometry()
        size = self.saved_size()
        if size:
            width = max(size[0], MIN_SIZE.width())
            height = max(size[1], MIN_SIZE.height())
        else:
            width = max(min(int(geo.width() * 0.5), 780), MIN_SIZE.width())
            height = MIN_SIZE.height()
        self.resize(width, height)
        if pos is not None:
            self.move(pos)
        else:
            # 預設放上方，與麥克風字幕（預設下方）天然分開
            self.move(geo.center().x() - width // 2, geo.top() + 64)
        self.apply_style()
        self.show()
        self.setWindowOpacity(self.target_opacity())

    def show_source(self, original: str):
        """原文先上、翻譯欄位顯示等待中；不進歷史（update_caption 才進）。"""
        self.original_label.setText(original)
        self.translated_label.setText("翻譯中…")

    def update_caption(self, original: str, translated: str):
        self.original_label.setText(original)
        self.translated_label.setText(translated)
        self._history.append((original, translated))
        if self.history_view.isVisible():
            self._refresh_history()

    def clear_history(self):
        self._history = []
        self.history_view.clear()

    def apply_style(self):
        size = int(self.config.get(
            self.CONFIG_SECTION, "font_size", default=20))
        original_font = QFont("Segoe UI")
        original_font.setPixelSize(max(12, round(size * 0.8)))
        self.original_label.setFont(original_font)
        self.original_label.setStyleSheet("color: #cfe9ef;")
        translated_font = QFont("Microsoft JhengHei")
        translated_font.setPixelSize(size)
        translated_font.setBold(True)
        self.translated_label.setFont(translated_font)
        self.translated_label.setStyleSheet("color: white;")
        self.update()

    # ---- 內部 ----

    def _toggle_history(self):
        if self.history_view.isVisible():
            self.history_view.hide()
        else:
            self._refresh_history()
            self.history_view.show()

    def _refresh_history(self):
        lines = []
        for original, translated in self._history:
            lines.append(f"{original}\n{translated}\n")
        self.history_view.setPlainText("\n".join(lines))
        self.history_view.moveCursor(QTextCursor.MoveOperation.End)

    def _on_close(self):
        self.hide()
        self.closed_by_user.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor(self.config.get(
            self.CONFIG_SECTION, "bg_color", default=DEFAULT_BG))
        bg.setAlpha(225)
        painter.setBrush(bg)
        painter.setPen(QColor(ACCENT))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 14, 14)
