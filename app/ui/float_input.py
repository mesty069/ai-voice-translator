from PySide6.QtCore import QEvent, QObject, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from qfluentwidgets import (
    FluentIcon,
    IndeterminateProgressRing,
    PrimaryPushButton,
    Theme,
    TransparentToolButton,
)

_FLAGS = (Qt.WindowType.FramelessWindowHint
          | Qt.WindowType.WindowStaysOnTopHint
          | Qt.WindowType.Tool)


class EnterSubmitFilter(QObject):
    """讓文字輸入框 Enter 直接送出、Ctrl+Enter 才換行。

    中文輸入法組字中的 Enter 由 IME 自己消化（是 InputMethod 事件），
    不會誤觸送出。"""

    def __init__(self, edit, submit):
        super().__init__(edit)
        self._edit = edit
        self._submit = submit
        edit.installEventFilter(self)

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)):
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._edit.textCursor().insertText("\n")
            else:
                self._submit()
            return True
        return super().eventFilter(obj, event)

BORDER = 8
_MAX = 16777215
MIN_SIZE = QSize(280, 160)
HEADER_HEIGHT = 34


class FloatingInputWidget(QWidget):
    """懸浮球模式下的浮動輸入框：輸入中文 → 翻譯（結果走字幕）。

    - 抓住頂端標題列拖曳移動；邊框拖曳調整大小（左右調寬、上下調高、
      角落全調）；位置與大小都會記住
    - 右上角 ✕ 關閉（同時把設定裡的開關關掉）
    - Ctrl+Enter 或「翻譯」按鈕送出
    """

    translate_requested = Signal(str)
    closed_by_user = Signal()

    def __init__(self, config):
        super().__init__(None, _FLAGS)
        self.config = config
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.setMinimumSize(MIN_SIZE)

        self._pressed = False
        self._moved = False
        self._resize_edges = None
        self._press_geo = None
        self._press_global = QPoint()
        self._drag_offset = QPoint()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 8, 14, 12)
        outer.setSpacing(6)

        header = QHBoxLayout()
        self.title_label = QLabel("輸入 → 翻譯", self)
        self.title_label.setStyleSheet(
            "color: #b8b8b8; font-size: 12px;"
            "font-family: 'Microsoft JhengHei';")
        header.addWidget(self.title_label)
        header.addStretch(1)
        # 深色底固定用白色圖示
        self.close_button = TransparentToolButton(
            FluentIcon.CLOSE.icon(Theme.DARK), self)
        self.close_button.setToolTip("關閉輸入框")
        self.close_button.setFixedSize(22, 22)
        self.close_button.clicked.connect(self._on_close)
        header.addWidget(self.close_button)
        outer.addLayout(header)

        self.input_edit = QTextEdit(self)
        self.input_edit.setPlaceholderText("輸入母語文字，Enter 翻譯（Ctrl+Enter 換行）")
        self.input_edit.setStyleSheet(
            "QTextEdit { background: rgba(255,255,255,18); color: white;"
            " border: 1px solid rgba(255,255,255,45); border-radius: 8px;"
            " font-size: 15px; font-family: 'Microsoft JhengHei';"
            " padding: 4px; }")
        outer.addWidget(self.input_edit, stretch=1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.busy_ring = IndeterminateProgressRing(self, start=True)
        self.busy_ring.setFixedSize(22, 22)
        self.busy_ring.setStrokeWidth(3)
        self.busy_ring.hide()
        button_row.addWidget(self.busy_ring)
        self.send_button = PrimaryPushButton("翻譯（Enter）", self)
        self.send_button.clicked.connect(self._submit)
        button_row.addWidget(self.send_button)
        outer.addLayout(button_row)

        EnterSubmitFilter(self.input_edit, self._submit)

    # ---- 對外 API ----

    def show_overlay(self, screen=None):
        pos = self._saved_pos()
        if pos is not None:
            screen = QApplication.screenAt(pos)
        screen = screen or QApplication.primaryScreen()
        geo = screen.availableGeometry()
        width = max(int(self.config.get("float_input", "width", default=340)),
                    MIN_SIZE.width())
        height = max(int(self.config.get("float_input", "height", default=190)),
                     MIN_SIZE.height())
        self.resize(width, height)
        self.apply_opacity()
        if pos is not None:
            self.move(pos)
        else:
            self.move(geo.right() - width - 24,
                      geo.bottom() - height - 140)
        self.show()

    def _saved_pos(self):
        x = self.config.get("float_input", "pos_x", default=None)
        y = self.config.get("float_input", "pos_y", default=None)
        if x is None or y is None:
            return None
        pos = QPoint(int(x), int(y))
        if QApplication.screenAt(pos) is None:
            return None
        return pos

    def apply_opacity(self):
        percent = self.config.get("float_input", "opacity", default=100)
        self.setWindowOpacity(max(0.3, min(1.0, int(percent) / 100)))

    def _on_close(self):
        self.hide()
        self.closed_by_user.emit()

    def set_busy(self, busy: bool):
        """等待 AI 翻譯時顯示旋轉圈並停用送出鈕。"""
        self.busy_ring.setVisible(busy)
        self.send_button.setEnabled(not busy)

    def _submit(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        self.set_busy(True)
        self.translate_requested.emit(text)
        self.input_edit.clear()

    # ---- 繪製 ----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(18, 18, 18, 230))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 12, 12)

    # ---- 拖曳移動（標題列）與邊框調整大小 ----

    def _edges_at(self, pos):
        left = pos.x() <= BORDER
        right = pos.x() >= self.width() - BORDER
        top = pos.y() <= BORDER
        bottom = pos.y() >= self.height() - BORDER
        if left or right or top or bottom:
            return (left, right, top, bottom)
        return None

    def _cursor_for(self, edges):
        left, right, top, bottom = edges
        if (left and top) or (right and bottom):
            return Qt.CursorShape.SizeFDiagCursor
        if (left and bottom) or (right and top):
            return Qt.CursorShape.SizeBDiagCursor
        if left or right:
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._pressed = True
        self._moved = False
        self._press_global = event.globalPosition().toPoint()
        self._press_geo = self.geometry()
        self._resize_edges = self._edges_at(event.position().toPoint())
        self._drag_offset = self._press_global - self.frameGeometry().topLeft()
        # 非邊框時只有標題列可拖曳（其餘區域是輸入區/按鈕）
        if not self._resize_edges and event.position().y() > HEADER_HEIGHT:
            self._pressed = False

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        gpos = event.globalPosition().toPoint()
        if not self._pressed:
            edges = self._edges_at(pos)
            self.setCursor(self._cursor_for(edges) if edges
                           else Qt.CursorShape.ArrowCursor)
            return
        self._moved = True
        if self._resize_edges:
            self._apply_resize(gpos)
        else:
            self.move(gpos - self._drag_offset)

    def _apply_resize(self, gpos):
        left, right, top, bottom = self._resize_edges
        geo = self._press_geo
        dx = gpos.x() - self._press_global.x()
        dy = gpos.y() - self._press_global.y()
        x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()
        if right:
            w = max(MIN_SIZE.width(), geo.width() + dx)
        if bottom:
            h = max(MIN_SIZE.height(), geo.height() + dy)
        if left:
            w = max(MIN_SIZE.width(), geo.width() - dx)
            x = geo.x() + geo.width() - w
        if top:
            h = max(MIN_SIZE.height(), geo.height() - dy)
            y = geo.y() + geo.height() - h
        self.setGeometry(x, y, w, h)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self._pressed:
            return
        self._pressed = False
        self._resize_edges = None
        if self._moved:
            self.config.set("float_input", "pos_x", self.pos().x())
            self.config.set("float_input", "pos_y", self.pos().y())
            self.config.set("float_input", "width", self.width())
            self.config.set("float_input", "height", self.height())
