from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

BUBBLE_SIZE = 56
ZONE_SIZE = 88
DRAG_THRESHOLD = 8

STATE_COLORS = {
    "idle": QColor("#0078d4"),
    "loading": QColor("#f39c12"),
    "recording": QColor("#e74c3c"),
    "processing": QColor("#3498db"),
    "error": QColor("#e74c3c"),
}

_FLAGS = (Qt.WindowType.FramelessWindowHint
          | Qt.WindowType.WindowStaysOnTopHint
          | Qt.WindowType.Tool)


class CloseZone(QWidget):
    """拖動懸浮球時出現在螢幕下方的關閉區，球拖進來放開就結束程式。"""

    def __init__(self):
        super().__init__(None, _FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(ZONE_SIZE, ZONE_SIZE)
        self._hovered = False

    def show_zone(self):
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.center().x() - ZONE_SIZE // 2,
                  screen.bottom() - ZONE_SIZE - 24)
        self._hovered = False
        self.show()

    def set_hovered(self, hovered: bool):
        if hovered != self._hovered:
            self._hovered = hovered
            self.update()

    def contains_global(self, point: QPoint) -> bool:
        return self.geometry().adjusted(-12, -12, 12, 12).contains(point)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#d13438") if self._hovered else QColor(60, 60, 60, 200)
        painter.setBrush(color)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
        margin = 4 if self._hovered else 10
        painter.drawEllipse(self.rect().adjusted(margin, margin, -margin, -margin))
        painter.setPen(QColor("white"))
        font = QFont()
        font.setPointSize(20 if self._hovered else 16)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "✕")


class BubbleWidget(QWidget):
    """Android 氣泡式懸浮球：點一下還原視窗、拖到關閉區結束程式。"""

    clicked = Signal()
    close_requested = Signal()

    def __init__(self):
        super().__init__(None, _FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(BUBBLE_SIZE, BUBBLE_SIZE)
        self.setToolTip("點一下開啟視窗，拖到下方 ✕ 關閉程式")
        self._color = STATE_COLORS["idle"]
        self._close_zone = CloseZone()
        self._pressed = False
        self._moved = False
        self._drag_offset = QPoint()
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - BUBBLE_SIZE - 12,
                  screen.top() + int(screen.height() * 0.35))

    def set_state(self, state: str):
        self._color = STATE_COLORS.get(state, STATE_COLORS["idle"])
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(self._color)
        painter.setPen(QPen(QColor(255, 255, 255, 230), 2))
        painter.drawEllipse(self.rect().adjusted(3, 3, -3, -3))
        font = QFont("Segoe UI Emoji")
        font.setPointSize(18)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "🎤")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self._moved = False
            self._drag_offset = (event.globalPosition().toPoint()
                                 - self.frameGeometry().topLeft())

    def mouseMoveEvent(self, event):
        if not self._pressed:
            return
        pos = event.globalPosition().toPoint()
        if not self._moved:
            start = self.frameGeometry().topLeft() + self._drag_offset
            if (pos - start).manhattanLength() < DRAG_THRESHOLD:
                return
            self._moved = True
            self._close_zone.show_zone()
        self.move(pos - self._drag_offset)
        self._close_zone.set_hovered(
            self._close_zone.contains_global(self.frameGeometry().center()))

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self._pressed:
            return
        self._pressed = False
        over_zone = (self._moved and self._close_zone.contains_global(
            self.frameGeometry().center()))
        self._close_zone.hide()
        if over_zone:
            self.close_requested.emit()
            return
        if not self._moved:
            self.clicked.emit()
            return
        self._snap_to_edge()

    def _snap_to_edge(self):
        screen = QApplication.primaryScreen().availableGeometry()
        geo = self.frameGeometry()
        y = min(max(geo.top(), screen.top() + 8),
                screen.bottom() - BUBBLE_SIZE - 8)
        if geo.center().x() < screen.center().x():
            self.move(screen.left() + 12, y)
        else:
            self.move(screen.right() - BUBBLE_SIZE - 12, y)

    def hideEvent(self, event):
        self._close_zone.hide()
        super().hideEvent(event)
