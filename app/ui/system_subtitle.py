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

from ..core.streaming_captions import DISPLAY_ROWS
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

        self.rows_layout = QVBoxLayout()
        self.rows_layout.setSpacing(4)
        outer.addLayout(self.rows_layout)
        self.row_widgets = []   # [(原文 QLabel, 翻譯 QLabel), ...]
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

    def preferred_height(self, rows: int) -> int:
        """放得下 rows 行（每行原文小字 + 翻譯大字）所需的高度。

        MIN_SIZE 的 130px 是為兩個 label 設計的，三行會被裁掉，所以依
        字級估：標頭 30 + 每行（0.8×字級 原文 + 字級 翻譯 + 行距 12）
        + 上下邊界 26，且不低於 MIN_SIZE。
        """
        size = int(self.config.get(
            self.CONFIG_SECTION, "font_size", default=20))
        per_row = max(12, round(size * 0.8)) + size + 12
        needed = 30 + max(0, int(rows)) * per_row + 26
        return max(MIN_SIZE.height(), needed)

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
            height = self.preferred_height(self.config.get(
                self.CONFIG_SECTION, "display_rows", default=DISPLAY_ROWS))
        self.resize(width, height)
        if pos is not None:
            self.move(pos)
        else:
            # 預設放上方，與麥克風字幕（預設下方）天然分開
            self.move(geo.center().x() - width // 2, geo.top() + 64)
        self.apply_style()
        self.show()
        self.setWindowOpacity(self.target_opacity())

    def set_rows(self, rows):
        """顯示最近幾行：每行原文小字 + 翻譯大字；未完句沒翻譯時顯示「…」。"""
        while len(self.row_widgets) < len(rows):
            original = QLabel("", self)
            original.setWordWrap(True)
            translated = QLabel("", self)
            translated.setWordWrap(True)
            self.rows_layout.addWidget(original)
            self.rows_layout.addWidget(translated)
            self.row_widgets.append((original, translated))
        while len(self.row_widgets) > len(rows):
            original, translated = self.row_widgets.pop()
            self.rows_layout.removeWidget(original)
            self.rows_layout.removeWidget(translated)
            original.deleteLater()
            translated.deleteLater()
        for (original, translated), row in zip(self.row_widgets, rows):
            original.setText(row.original)
            translated.setText(row.translated or ("…" if not row.is_final else ""))
        self.apply_style()   # 先套字型，量高度才準
        self._fit_height(len(rows))

    def required_height(self, rows: int) -> int:
        """目前寬度下真正放得下所有文字（含換行）的高度。

        preferred_height 只按行數估；長句換行後會更高。不能問 layout 的
        heightForWidth：剛 addWidget 完它的快取還沒更新（回 -1），要等下一輪
        事件迴圈；QLabel 自己的 heightForWidth 則立刻就準，所以逐個加總。
        邊界 20/10/12/16 與行距 6/4 對應 __init__ 裡的 layout 設定。
        """
        margins = self.layout().contentsMargins()
        label_width = max(50, self.width() - margins.left() - margins.right())
        header_h = max(22, self.tag_label.sizeHint().height())
        labels = [lbl for pair in self.row_widgets for lbl in pair]
        text_h = sum(lbl.heightForWidth(label_width) for lbl in labels)
        gaps = self.rows_layout.spacing() * max(0, len(labels) - 1)
        needed = (margins.top() + header_h + self.layout().spacing()
                  + text_h + gaps + margins.bottom())
        return max(self.preferred_height(rows), needed)

    def _fit_height(self, rows: int):
        """把最小高度設成實際需求：Qt 會自動把視窗撐開，文字不會被裁掉；
        使用者拉大過的高度照樣尊重（只設下限，不主動縮小）。"""
        needed = self.required_height(rows)
        self.setMinimumHeight(needed)
        if self.height() < needed:
            self.resize(self.width(), needed)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 拉窄後文字會多折幾行，需求高度變了要重算
        if event.oldSize().width() != event.size().width() and self.row_widgets:
            self._fit_height(len(self.row_widgets))

    def add_history(self, original: str, translated: str):
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
        translated_font = QFont("Microsoft JhengHei")
        translated_font.setPixelSize(size)
        translated_font.setBold(True)
        last = len(self.row_widgets) - 1
        for i, (original, translated) in enumerate(self.row_widgets):
            original.setFont(original_font)
            translated.setFont(translated_font)
            # 舊句淡一點，正在講的那句最亮
            dim = i < last
            original.setStyleSheet("color: #8fb8c2;" if dim else "color: #cfe9ef;")
            translated.setStyleSheet("color: #d0d0d0;" if dim else "color: white;")
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
