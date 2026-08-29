from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

_FLAGS = (Qt.WindowType.FramelessWindowHint
          | Qt.WindowType.WindowStaysOnTopHint
          | Qt.WindowType.Tool
          | Qt.WindowType.WindowDoesNotAcceptFocus)


class WaitHintOverlay(QWidget):
    """螢幕正中央的「請稍等再說」提示框。

    按住錄音鍵後要等其他軟體切換麥克風（isolation_settle_ms），
    這段期間講話會漏音，所以顯示明顯提示；可以開口時立刻消失。
    半透明深色底＋粗白字，不搶焦點、滑鼠穿透。
    """

    def __init__(self):
        super().__init__(None, _FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(56, 30, 56, 30)
        layout.setSpacing(8)

        self.title_label = QLabel("請稍等再說…", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Microsoft JhengHei")
        title_font.setPixelSize(34)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: white;")
        layout.addWidget(self.title_label)

        self.detail_label = QLabel("正在切斷其他軟體的麥克風", self)
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail_font = QFont("Microsoft JhengHei")
        detail_font.setPixelSize(16)
        self.detail_label.setFont(detail_font)
        self.detail_label.setStyleSheet("color: #ffd75e;")
        layout.addWidget(self.detail_label)

    def show_centered(self, screen=None):
        screen = screen or QApplication.primaryScreen()
        geo = screen.availableGeometry()
        self.adjustSize()
        self.move(geo.center().x() - self.width() // 2,
                  geo.center().y() - self.height() // 2)
        # 不用淡入：提示只在幾百毫秒的空窗期出現，要立刻看得到
        self.show()
        self.raise_()

    def set_visible(self, visible: bool, screen=None):
        if visible:
            self.show_centered(screen)
        else:
            self.hide()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(16, 16, 16, 205))          # 半透明深底
        painter.setPen(QPen(QColor(255, 215, 94, 220), 2))  # 琥珀色細框，醒目
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 16, 16)
