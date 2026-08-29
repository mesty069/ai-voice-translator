from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

LONG_PRESS_MS = 500

BUBBLE_SIZE = 84   # 加大＋光暈，在滿版遊戲/影片旁也找得到
ZONE_SIZE = 88
DRAG_THRESHOLD = 8

STATE_COLORS = {
    "idle": QColor("#0078d4"),
    "loading": QColor("#0078d4"),  # 載入中也能直接用，不再顯示金色提示
    "recording": QColor("#e74c3c"),
    "processing": QColor("#3498db"),
    "error": QColor("#e74c3c"),
}

_FLAGS = (Qt.WindowType.FramelessWindowHint
          | Qt.WindowType.WindowStaysOnTopHint
          | Qt.WindowType.Tool)


def _blend(a: QColor, b: QColor, t: float) -> QColor:
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
        round(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


class CloseZone(QWidget):
    """拖動懸浮球時出現在螢幕下方的關閉區，球拖進來放開就結束程式。"""

    def __init__(self):
        super().__init__(None, _FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(ZONE_SIZE, ZONE_SIZE)
        self._hover_t = 0.0     # 0=普通 1=吸附放大
        self._hovered = False
        self._screen = None

        self._hover_anim = QVariantAnimation(self)
        self._hover_anim.setDuration(140)
        self._hover_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._hover_anim.valueChanged.connect(self._on_hover_t)

        self._slide_anim = QPropertyAnimation(self, b"pos", self)
        self._slide_anim.setDuration(220)
        self._slide_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._fade_anim = QVariantAnimation(self)
        self._fade_anim.setDuration(220)
        self._fade_anim.valueChanged.connect(
            lambda v: self.setWindowOpacity(v))
        self._fade_anim.finished.connect(self._after_fade)

    def _on_hover_t(self, value):
        self._hover_t = value
        self.update()

    def _home_pos(self) -> QPoint:
        screen = (self._screen or QApplication.primaryScreen()).availableGeometry()
        return QPoint(screen.center().x() - ZONE_SIZE // 2,
                      screen.bottom() - ZONE_SIZE - 24)

    def show_zone(self, screen=None):
        """在指定螢幕（預設主螢幕）從下緣滑入＋淡入。"""
        self._screen = screen
        self._hovered = False
        self._hover_t = 0.0
        home = self._home_pos()
        self.move(home + QPoint(0, 48))
        self.setWindowOpacity(0.0)
        self.show()
        self._slide_anim.stop()
        self._slide_anim.setStartValue(self.pos())
        self._slide_anim.setEndValue(home)
        self._slide_anim.start()
        self._fade_anim.stop()
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()

    def hide_zone(self):
        """往下滑出＋淡出後隱藏。"""
        if not self.isVisible():
            return
        self._slide_anim.stop()
        self._slide_anim.setStartValue(self.pos())
        self._slide_anim.setEndValue(self._home_pos() + QPoint(0, 48))
        self._slide_anim.start()
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self.windowOpacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.start()

    def _after_fade(self):
        # 淡出結束才隱藏；淡入結束（不透明度 1）什麼都不做
        if self.windowOpacity() < 0.01:
            self.hide()

    def set_hovered(self, hovered: bool):
        if hovered == self._hovered:
            return
        self._hovered = hovered
        self._hover_anim.stop()
        self._hover_anim.setStartValue(self._hover_t)
        self._hover_anim.setEndValue(1.0 if hovered else 0.0)
        self._hover_anim.start()

    def contains_global(self, point: QPoint) -> bool:
        return self.geometry().adjusted(-12, -12, 12, 12).contains(point)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = self._hover_t
        color = _blend(QColor(60, 60, 60, 200), QColor("#d13438"), t)
        painter.setBrush(color)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
        margin = round(10 - 6 * t)
        painter.drawEllipse(self.rect().adjusted(margin, margin, -margin, -margin))
        painter.setPen(QColor("white"))
        font = QFont()
        font.setPointSizeF(16 + 4 * t)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "✕")


class BubbleMenu(QWidget):
    """長按懸浮球展開的功能選單。"""

    input_clicked = Signal()
    window_clicked = Signal()
    quit_clicked = Signal()

    def __init__(self):
        super().__init__(None, _FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)
        self.setStyleSheet(
            "QPushButton { color: white; background: transparent;"
            " border: none; border-radius: 8px; padding: 8px 16px;"
            " font-size: 14px; font-family: 'Microsoft JhengHei';"
            " text-align: left; }"
            "QPushButton:hover { background: rgba(255,255,255,36); }")
        for text, signal in (
                ("⌨  輸入框 開/關", self.input_clicked),
                ("🗖  開啟主視窗", self.window_clicked),
                ("✕  結束程式", self.quit_clicked)):
            btn = QPushButton(text, self)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(signal)
            btn.clicked.connect(self.hide_menu)
            layout.addWidget(btn)

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(150)
        self._fade.finished.connect(self._after_fade)
        self._auto_hide = QTimer(self)
        self._auto_hide.setSingleShot(True)
        self._auto_hide.setInterval(5000)
        self._auto_hide.timeout.connect(self.hide_menu)

    def _position_near(self, bubble: QWidget):
        geo = bubble.frameGeometry()
        screen = (QApplication.screenAt(geo.center())
                  or QApplication.primaryScreen()).availableGeometry()
        # 靠螢幕右半 → 選單開在球的左邊，反之開右邊
        if geo.center().x() > screen.center().x():
            x = geo.left() - self.width() - 8
        else:
            x = geo.right() + 8
        y = min(max(geo.center().y() - self.height() // 2, screen.top() + 8),
                screen.bottom() - self.height() - 8)
        self.move(x, y)

    def follow(self, bubble: QWidget):
        """球被拖動時讓選單跟著走。"""
        if not self.isVisible():
            return
        self._position_near(bubble)
        self._auto_hide.start()  # 拖動中重新計時，不要拖到一半消失

    def show_near(self, bubble: QWidget):
        self.adjustSize()
        self._position_near(bubble)
        self.setWindowOpacity(0.0)
        self.show()
        self._fade.stop()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()
        self._auto_hide.start()

    def hide_menu(self):
        if not self.isVisible():
            return
        self._auto_hide.stop()
        self._fade.stop()
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _after_fade(self):
        if self.windowOpacity() < 0.01:
            self.hide()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(24, 24, 24, 235))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 12, 12)


class BubbleWidget(QWidget):
    """Android 氣泡式懸浮球：點一下還原視窗、拖到關閉區結束程式。

    所有互動都有動畫：彈出/收起、hover、按壓、貼邊、狀態色漸變、錄音呼吸。
    """

    clicked = Signal()
    close_requested = Signal()
    input_toggle_requested = Signal()

    def __init__(self):
        super().__init__(None, _FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(BUBBLE_SIZE, BUBBLE_SIZE)
        self.setToolTip("點一下開啟視窗，長按展開選單，拖到下方 ✕ 關閉程式")
        self.menu = BubbleMenu()
        self.menu.input_clicked.connect(self.input_toggle_requested)
        self.menu.window_clicked.connect(self.clicked)
        self.menu.quit_clicked.connect(self.close_requested)
        self._menu_shown_this_press = False
        self._long_press_timer = QTimer(self)
        self._long_press_timer.setSingleShot(True)
        self._long_press_timer.setInterval(LONG_PRESS_MS)
        self._long_press_timer.timeout.connect(self._on_long_press)
        self._color = QColor(STATE_COLORS["idle"])
        self._target_color = QColor(self._color)
        self._state = "idle"
        self._close_zone = CloseZone()
        self._pressed = False
        self._moved = False
        self._hiding = False     # 縮小消失動畫中，忽略滑鼠互動以免取消回呼
        self._drag_offset = QPoint()
        self._scale = 1.0        # 繪製縮放（彈出/hover/按壓共用）
        self._pulse_t = -1.0     # 錄音呼吸 0~1；<0 表示不畫

        self._scale_anim = QVariantAnimation(self)
        self._scale_anim.setDuration(160)
        self._scale_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._scale_anim.valueChanged.connect(self._on_scale)
        self._scale_anim.finished.connect(self._on_scale_done)
        self._scale_done_cb = None

        self._color_anim = QVariantAnimation(self)
        self._color_anim.setDuration(220)
        self._color_anim.valueChanged.connect(self._on_color_t)

        self._pulse_anim = QVariantAnimation(self)
        self._pulse_anim.setDuration(1100)
        self._pulse_anim.setStartValue(0.0)
        self._pulse_anim.setEndValue(1.0)
        self._pulse_anim.setLoopCount(-1)
        self._pulse_anim.valueChanged.connect(self._on_pulse)

        self._pos_anim = QPropertyAnimation(self, b"pos", self)
        self._pos_anim.setDuration(260)
        self._pos_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        # 放開後貼回螢幕內的動畫，選單也要同步跟著
        self._pos_anim.valueChanged.connect(
            lambda _: self.menu.follow(self))

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - BUBBLE_SIZE - 12,
                  screen.top() + int(screen.height() * 0.35))

    # ---- 動畫輔助 ----

    def _on_scale(self, value):
        self._scale = value
        self.update()

    def _on_color_t(self, t):
        self._color = _blend(self._from_color, self._target_color, t)
        self.update()

    def _on_pulse(self, t):
        self._pulse_t = t
        self.update()

    def _on_scale_done(self):
        cb, self._scale_done_cb = self._scale_done_cb, None
        if cb is not None:
            cb()

    def _animate_scale(self, target: float, duration: int = 160,
                       easing=QEasingCurve.Type.OutCubic, on_done=None):
        # 先清回呼再 stop：stop() 會同步發出 finished，不能誤觸發舊回呼
        self._scale_done_cb = None
        self._scale_anim.stop()
        self._scale_done_cb = on_done
        self._scale_anim.setDuration(duration)
        self._scale_anim.setEasingCurve(easing)
        self._scale_anim.setStartValue(self._scale)
        self._scale_anim.setEndValue(target)
        self._scale_anim.start()

    # ---- 對外 API ----

    def show_animated(self):
        """從 0 彈出（帶回彈感），接著放大縮小脈衝幾下吸引注意。"""
        self._scale = 0.0
        self.show()
        self._animate_scale(1.0, duration=260,
                            easing=QEasingCurve.Type.OutBack,
                            on_done=lambda: self._attention_pulse(4))

    def _attention_pulse(self, remaining: int):
        """出現後的注意力脈衝：放大→縮回，共 remaining/2 輪。
        使用者一碰（hover/按壓）或開始隱藏就中止。"""
        if remaining <= 0 or self._pressed or self._hiding:
            if not self._pressed and not self._hiding:
                self._animate_scale(1.0)
            return
        target = 1.18 if remaining % 2 == 0 else 1.0
        self._animate_scale(
            target, duration=340,
            easing=QEasingCurve.Type.InOutQuad,
            on_done=lambda: self._attention_pulse(remaining - 1))

    def hide_animated(self, on_done=None):
        """縮小消失，結束後呼叫 on_done。"""
        if self._hiding:
            return
        self._hiding = True

        def _finish():
            self._hiding = False
            self.hide()
            self._scale = 1.0
            if on_done is not None:
                on_done()
        self._animate_scale(0.0, duration=180,
                            easing=QEasingCurve.Type.InBack, on_done=_finish)

    def set_state(self, state: str):
        if state == self._state:
            return
        self._state = state
        self._from_color = QColor(self._color)
        self._target_color = QColor(
            STATE_COLORS.get(state, STATE_COLORS["idle"]))
        self._color_anim.stop()
        self._color_anim.setStartValue(0.0)
        self._color_anim.setEndValue(1.0)
        self._color_anim.start()
        if state == "recording":
            self._pulse_anim.start()
        else:
            self._pulse_anim.stop()
            self._pulse_t = -1.0
            self.update()

    # ---- 繪製 ----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = self.rect().center()
        painter.translate(center)
        painter.scale(max(self._scale, 0.0), max(self._scale, 0.0))
        painter.translate(-center.x(), -center.y())

        # 外圈光暈：讓球在任何背景上都醒目
        glow = QColor(self._color)
        glow.setAlpha(70)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(self.rect().adjusted(1, 1, -1, -1))

        # 錄音呼吸圈：由球邊緣向外擴散並淡出
        if self._pulse_t >= 0.0:
            t = self._pulse_t
            alpha = int(180 * (1.0 - t))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(
                self._color.red(), self._color.green(),
                self._color.blue(), alpha), 3))
            m = round(7 - 6 * t)
            painter.drawEllipse(self.rect().adjusted(m, m, -m, -m))

        # 本體：狀態色的放射漸層（中心亮、邊緣深）＋粗白框
        from PySide6.QtGui import QRadialGradient
        radius = (self.width() - 16) / 2
        grad = QRadialGradient(center.x() - radius * 0.3,
                               center.y() - radius * 0.3, radius * 1.8)
        grad.setColorAt(0.0, self._color.lighter(145))
        grad.setColorAt(1.0, self._color.darker(110))
        painter.setBrush(grad)
        painter.setPen(QPen(QColor(255, 255, 255, 245), 3))
        painter.drawEllipse(self.rect().adjusted(8, 8, -8, -8))
        font = QFont("Segoe UI Emoji")
        font.setPointSize(24)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "🎤")

    # ---- 互動 ----

    def enterEvent(self, event):
        if (not self._pressed and not self._hiding
                and self._scale_anim.state() != QVariantAnimation.State.Running):
            self._animate_scale(1.1)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self._pressed and not self._hiding:
            self._animate_scale(1.0)
        super().leaveEvent(event)

    def _on_long_press(self):
        if self._pressed and not self._moved:
            self._menu_shown_this_press = True
            self.menu.show_near(self)

    def mousePressEvent(self, event):
        if self._hiding:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.menu.hide_menu()
            self._pressed = True
            self._moved = False
            self._menu_shown_this_press = False
            self._long_press_timer.start()
            self._drag_offset = (event.globalPosition().toPoint()
                                 - self.frameGeometry().topLeft())
            self._animate_scale(0.9, duration=100)

    def mouseMoveEvent(self, event):
        if not self._pressed:
            return
        pos = event.globalPosition().toPoint()
        if not self._moved:
            start = self.frameGeometry().topLeft() + self._drag_offset
            if (pos - start).manhattanLength() < DRAG_THRESHOLD:
                return
            self._moved = True
            self._long_press_timer.stop()
            self._close_zone.show_zone(self.screen())
        self.move(pos - self._drag_offset)
        self.menu.follow(self)
        over = self._close_zone.contains_global(self.frameGeometry().center())
        self._close_zone.set_hovered(over)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self._pressed:
            return
        self._pressed = False
        self._long_press_timer.stop()
        over_zone = (self._moved and self._close_zone.contains_global(
            self.frameGeometry().center()))
        self._close_zone.hide_zone()
        if over_zone:
            self.close_requested.emit()
            return
        if not self._moved:
            if not self._menu_shown_this_press:  # 長按已開選單就不觸發還原
                self.clicked.emit()
            else:
                self._animate_scale(1.0)
            return
        self._animate_scale(1.0)
        self._clamp_to_screen()

    def _clamp_to_screen(self):
        """自由放置：只有超出目前所在螢幕邊界時才滑回可見範圍。"""
        geo = self.frameGeometry()
        screen_obj = QApplication.screenAt(geo.center()) or self.screen() \
            or QApplication.primaryScreen()
        screen = screen_obj.availableGeometry()
        x = min(max(geo.left(), screen.left() + 8),
                screen.right() - BUBBLE_SIZE - 8)
        y = min(max(geo.top(), screen.top() + 8),
                screen.bottom() - BUBBLE_SIZE - 8)
        target = QPoint(x, y)
        if target == self.pos():
            return
        self._pos_anim.stop()
        self._pos_anim.setStartValue(self.pos())
        self._pos_anim.setEndValue(target)
        self._pos_anim.start()

    def hideEvent(self, event):
        self._close_zone.hide()
        self.menu.hide()
        super().hideEvent(event)
