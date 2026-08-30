import html

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

from ..core.streaming_captions import (
    DISPLAY_ROWS,
    VERYLONG_THRESHOLD,
    join_words,
    shorten_display_sentence,
)
from .overlay_base import DraggableResizableOverlay

_FLAGS = (Qt.WindowType.FramelessWindowHint
          | Qt.WindowType.WindowStaysOnTopHint
          | Qt.WindowType.Tool
          | Qt.WindowType.WindowDoesNotAcceptFocus)

DEFAULT_BG = "#0d2b33"
ACCENT = "#4dd0e1"      # 青色，與麥克風字幕（深灰白字）明顯區分
DIM_TEXT = "#d0d0d0"    # 前幾句翻譯（照 OverlayWindow.xaml）
ACTIVE_TEXT = "#ffffff"  # 正在講的那句翻譯
ORIGINAL_TEXT = "#cfe9ef"   # 下方當前句原文
MIN_SIZE = QSize(360, 130)


def _separator(left: str, right: str) -> str:
    """兩段文字之間該不該有空白（沿用 join_words 的中日韓規則）。"""
    if not left or not right:
        return ""
    return join_words([left, right])[len(left):-len(right)]


class SystemSubtitleOverlay(DraggableResizableOverlay):
    """系統聲音的雙語字幕：上方翻譯段落、下方當前句原文（照 OverlayWindow.xaml）。

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
        self._rows_shown = 0    # 上次 set_rows 的列數（重算高度用）

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
        # 上：翻譯段落（前幾句較淡 + 當前句白色，rich text 上色）
        self.translation_label = QLabel("", self)
        self.translation_label.setWordWrap(True)
        self.translation_label.setTextFormat(Qt.TextFormat.RichText)
        self.rows_layout.addWidget(self.translation_label)
        # 下：當前句原文
        self.original_label = QLabel("", self)
        self.original_label.setWordWrap(True)
        # 原文可能長得像 HTML（<b>、&），一律當純文字顯示
        self.original_label.setTextFormat(Qt.TextFormat.PlainText)
        self.rows_layout.addWidget(self.original_label)
        outer.addLayout(self.rows_layout)
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
        """放得下 rows 句（翻譯大字 + 原文小字）所需的高度估值。

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
            height = self.preferred_height(1 + int(self.config.get(
                self.CONFIG_SECTION, "display_rows", default=DISPLAY_ROWS)))
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
        """上方把所有翻譯接成一段（前幾句較淡、當前句白色），下方放當前句原文。

        當前句還沒翻好時就只顯示前幾句，不補「…」（照 LiveCaptions-Translator：
        翻譯是一段連續的文字，佔位符會讓它一直跳動）。
        """
        self._rows_shown = len(rows)
        if not rows:
            self.translation_label.setText("")
            self.original_label.setText("")
            self.apply_style()
            self._fit_height(0)
            return
        previous = join_words([r.translated for r in rows[:-1] if r.translated])
        current = rows[-1].translated
        parts = []
        if previous:
            parts.append(f'<span style="color: {DIM_TEXT}">'
                         f'{html.escape(previous)}</span>')
        if current:
            parts.append(_separator(previous, current))
            parts.append(f'<span style="color: {ACTIVE_TEXT}">'
                         f'{html.escape(current)}</span>')
        self.translation_label.setText("".join(parts))
        self.original_label.setText(
            shorten_display_sentence(rows[-1].original, VERYLONG_THRESHOLD))
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
        labels = [self.translation_label, self.original_label]
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
        if (event.oldSize().width() != event.size().width()
                and getattr(self, "_rows_shown", 0)):
            self._fit_height(self._rows_shown)

    def add_history(self, original: str, translated: str):
        # 短句併進前一句時會再送一次「合併後的整句」：直接取代上一筆，
        # 逐字稿才不會同一段話出現兩次
        last = self._history[-1][0] if self._history else None
        if last is not None and original != last and original.startswith(last):
            self._history[-1] = (original, translated)
        else:
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
        self.translation_label.setFont(translated_font)
        # 段落裡各句的顏色由 set_rows 的 <span> 決定，這裡只給預設色
        self.translation_label.setStyleSheet(f"color: {ACTIVE_TEXT};")
        self.original_label.setFont(original_font)
        self.original_label.setStyleSheet(f"color: {ORIGINAL_TEXT};")
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
