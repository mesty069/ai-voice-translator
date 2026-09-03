from PySide6.QtCore import QPropertyAnimation, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from qfluentwidgets import (
    FlowLayout,
    FluentIcon,
    IndeterminateProgressRing,
    Slider,
    Theme,
    TransparentToolButton,
)

from ..core import tts
from .float_input import EnterSubmitFilter
from .overlay_base import DraggableResizableOverlay

_FLAGS = (Qt.WindowType.FramelessWindowHint
          | Qt.WindowType.WindowStaysOnTopHint
          | Qt.WindowType.Tool
          | Qt.WindowType.WindowDoesNotAcceptFocus)

MESSAGE_DURATION_MS = 6000
DEFAULT_DURATION_SECONDS = 10
DEFAULT_FONT_SIZE = 21
DEFAULT_BG_COLOR = "#121212"
DEFAULT_FONT_COLOR = "#ffffff"
_MAX = 16777215

FONT_WEIGHTS = {
    "light": QFont.Weight.Light,
    "normal": QFont.Weight.Normal,
    "medium": QFont.Weight.Medium,
    "demibold": QFont.Weight.DemiBold,
    "bold": QFont.Weight.Bold,
    "black": QFont.Weight.Black,
}

CHIP_H_PAD = 7      # 方塊左右內距（sizeHint 自己算，才不會吃到 QPushButton 的最小寬）
CHIP_V_PAD = 3
CHIP_SPACING = 5


class WordChip(QPushButton):
    """英文行的一個單字方塊：點下去從這個字開始往後念。"""

    def __init__(self, word: str, index: int, parent=None):
        super().__init__(word, parent)
        self.index = index
        self._css = ""
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip("從這個字開始往後念")

    def apply_look(self, font: QFont, color: QColor):
        self.setFont(font)
        r, g, b = color.red(), color.green(), color.blue()
        css = ("QPushButton {"
               f" color: rgb({r},{g},{b}); background: rgba(255,255,255,26);"
               " border: none; border-radius: 6px; }"
               "QPushButton:hover { background: rgba(255,255,255,60); }"
               "QPushButton:pressed { background: rgba(255,255,255,90); }")
        if css != self._css:   # 縮字級時會重跑好幾輪，樣式沒變就別重設
            self._css = css
            self.setStyleSheet(css)
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        metrics = self.fontMetrics()
        return QSize(metrics.horizontalAdvance(self.text()) + CHIP_H_PAD * 2,
                     metrics.height() + CHIP_V_PAD * 2)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()


class WordFlow(QWidget):
    """裝一排單字方塊的容器（FlowLayout 自動換行）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flow = FlowLayout(self, needAni=False)
        self.flow.setContentsMargins(0, 0, 0, 0)
        self.flow.setHorizontalSpacing(CHIP_SPACING)
        self.flow.setVerticalSpacing(CHIP_SPACING)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self.flow.heightForWidth(max(1, width))

    def sizeHint(self) -> QSize:
        width = self.width() or 400
        return QSize(width, self.heightForWidth(width))


class SubtitleOverlay(DraggableResizableOverlay):
    """懸浮球模式下的字幕：顯示梳理後中文＋英文翻譯，數秒後淡出。

    - 顯示秒數與字體大小依 config 的 subtitle 區段
    - 英文行是一排單字方塊，點某個字就從那個字開始往後念
    - ⏸ 暫停/繼續朗讀；🔊 重新播放英文語音；右上角 ✕ 關閉
    - 語速、音量拖桿即時寫 config（與設定頁雙向同步）
    - 內部拖曳移動位置；邊框拖曳調整大小（左右調寬、上下調高、角落全調），
      最小尺寸隨字體大小連動；位置與大小都會記住
    - 不搶焦點（遊戲/通話中不會被切走）
    """

    CONFIG_SECTION = "subtitle"

    replay_requested = Signal()
    rate_changed = Signal(int)          # 朗讀語速（與設定頁同一個值）
    volume_changed = Signal(int)        # 朗讀音量（與設定頁同一個值）
    retranslate_requested = Signal(str)  # 使用者改完中文 → 重新翻譯
    speak_requested = Signal(str)       # 點單字方塊 → 從那個字開始念

    def __init__(self, config):
        super().__init__(config, _FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(28, 20, 16, 18)
        outer.setSpacing(10)

        text_col = QVBoxLayout()
        text_col.setSpacing(6)
        self.zh_label = QLabel(self)
        self.zh_label.setWordWrap(True)
        self.zh_label.setCursor(Qt.CursorShape.IBeamCursor)
        self.zh_label.setToolTip("點一下修改原文，Enter 或點到別處會重新翻譯")
        # 英文行：一排可點的單字方塊（FlowLayout 自動換行）
        self.words_widget = WordFlow(self)
        self._chips: list[WordChip] = []
        self._words: list[str] = []

        # 中文行的就地編輯器（平常隱藏，點中文行才出現）
        self.zh_edit = QTextEdit(self)
        self.zh_edit.hide()
        self.zh_edit.setStyleSheet(
            "QTextEdit { background: rgba(255,255,255,24); color: white;"
            " border: 1px solid rgba(255,255,255,70); border-radius: 6px;"
            " padding: 2px; }")
        EnterSubmitFilter(self.zh_edit, self._finish_edit)
        self.zh_edit.focusOutEvent = self._on_edit_focus_out
        self._editing = False
        self._reading = False          # 朗讀中（倒數暫停）
        self._paused = False           # 朗讀暫停中（倒數維持暫停）
        self.last_edit_finished = 0.0  # 供全域 Enter 冷卻判斷
        text_col.addWidget(self.zh_label)
        text_col.addWidget(self.words_widget)
        text_col.addStretch(1)
        outer.addLayout(text_col, stretch=1)

        # 右側控制欄：等待圈 + 🔊 重播 + 語速拖桿
        control_col = QVBoxLayout()
        control_col.setSpacing(2)
        control_col.addStretch(1)
        self.busy_ring = IndeterminateProgressRing(self, start=True)
        self.busy_ring.setFixedSize(24, 24)
        self.busy_ring.setStrokeWidth(3)
        self.busy_ring.hide()
        control_col.addWidget(self.busy_ring,
                              alignment=Qt.AlignmentFlag.AlignHCenter)
        # 字幕底永遠是深色 → 圖示固定用深色主題的白色版本，亮色模式也看得清
        self.pause_button = TransparentToolButton(
            FluentIcon.PAUSE.icon(Theme.DARK), self)
        self.pause_button.setToolTip("暫停朗讀")
        self.pause_button.setFixedSize(48, 48)
        self.pause_button.setIconSize(QSize(24, 24))
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._on_pause_clicked)
        control_col.addWidget(self.pause_button,
                              alignment=Qt.AlignmentFlag.AlignHCenter)
        self.replay_button = TransparentToolButton(
            FluentIcon.VOLUME.icon(Theme.DARK), self)
        self.replay_button.setToolTip("重新播放英文語音")
        self.replay_button.setFixedSize(36, 36)
        self.replay_button.clicked.connect(self._on_replay)
        control_col.addWidget(self.replay_button,
                              alignment=Qt.AlignmentFlag.AlignHCenter)
        self.rate_slider = Slider(Qt.Orientation.Horizontal, self)
        self.rate_slider.setRange(80, 350)
        self.rate_slider.setFixedWidth(92)
        self.rate_slider.setToolTip("拖動調整朗讀語速")
        self.rate_slider.valueChanged.connect(self._on_rate_changed)
        self.rate_slider.sliderPressed.connect(self._on_rate_drag_start)
        self.rate_slider.sliderReleased.connect(self._on_rate_drag_end)
        control_col.addWidget(self.rate_slider)
        self.rate_label = QLabel("", self)
        self.rate_label.setStyleSheet("color: #a8a8a8; font-size: 11px;")
        self.rate_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        control_col.addWidget(self.rate_label)
        self.volume_slider = Slider(Qt.Orientation.Horizontal, self)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setFixedWidth(92)
        self.volume_slider.setToolTip("拖動調整朗讀音量")
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        self.volume_slider.sliderPressed.connect(self._on_rate_drag_start)
        self.volume_slider.sliderReleased.connect(self._on_volume_drag_end)
        control_col.addWidget(self.volume_slider)
        self.volume_label = QLabel("", self)
        self.volume_label.setStyleSheet("color: #a8a8a8; font-size: 11px;")
        self.volume_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        control_col.addWidget(self.volume_label)
        control_col.addStretch(1)
        outer.addLayout(control_col)

        self.close_button = TransparentToolButton(
            FluentIcon.CLOSE.icon(Theme.DARK), self)
        self.close_button.setToolTip("關閉字幕")
        self.close_button.setFixedSize(24, 24)
        self.close_button.clicked.connect(self.dismiss)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out)

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(200)
        self._fade.finished.connect(self._after_fade)

    # ---- 對外 API ----

    def show_result(self, refined: str, english: str, screen=None):
        self._set_lines(refined, english)
        self._set_controls_visible(bool(english))
        self._sync_rate_from_config()
        self._sync_volume_from_config()
        self._popup(screen, self._duration_ms())

    def show_message(self, text: str, screen=None):
        self._set_lines(text, "")
        self._set_controls_visible(False)
        self._popup(screen, MESSAGE_DURATION_MS)

    def set_reading(self, reading: bool):
        """朗讀中暫停倒數；播完（含重播播完）重新從頭倒數。

        暫停時 controller 不會發結束訊號，所以這裡不會被叫成 False，
        倒數自然維持暫停；新一句開始朗讀則把暫停狀態清掉。"""
        self._reading = reading
        self._paused = False
        self._update_pause_button()
        if reading:
            self._hide_timer.stop()
        elif (self.isVisible() and not self._editing
                and not self.busy_ring.isVisible()):
            self._hide_timer.start(self._duration_ms())

    # ---- 單字方塊 ----

    def word_chips(self) -> list:
        return list(self._chips)

    @property
    def is_paused(self) -> bool:
        return self._paused

    def _set_words(self, english: str):
        self._words = english.split()
        while self._chips:
            chip = self._chips.pop()
            self.words_widget.flow.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        for index, word in enumerate(self._words):
            chip = WordChip(word, index, self.words_widget)
            chip.clicked.connect(
                lambda _=False, i=index: self._on_chip_clicked(i))
            self.words_widget.flow.addWidget(chip)
            chip.show()
            self._chips.append(chip)
        self.words_widget.setVisible(bool(self._chips))

    def _on_chip_clicked(self, index: int):
        """從這個字開始往後念（沿用現有裝置/語速/音量設定）。"""
        if not (0 <= index < len(self._words)):
            return
        # 點字算「重新朗讀」：倒數重算（真正的暫停由 set_reading 接手）
        self._hide_timer.start(self._duration_ms())
        self._paused = False
        self._update_pause_button()
        self.speak_requested.emit(" ".join(self._words[index:]))

    def _set_controls_visible(self, visible: bool):
        for widget in (self.pause_button, self.replay_button,
                       self.rate_slider, self.rate_label,
                       self.volume_slider, self.volume_label):
            widget.setVisible(visible)

    # ---- 暫停/繼續 ----

    def _on_pause_clicked(self):
        """暫停/繼續：只設旗標給 TTS worker，GUI 執行緒不等它。"""
        if self._paused:
            if tts.resume(self._volume_fraction()):
                self._paused = False
        elif tts.pause():
            self._paused = True
        self._update_pause_button()

    def _update_pause_button(self):
        icon = FluentIcon.PLAY_SOLID if self._paused else FluentIcon.PAUSE
        self.pause_button.setIcon(icon.icon(Theme.DARK))
        self.pause_button.setToolTip("繼續朗讀" if self._paused else "暫停朗讀")
        self.pause_button.setEnabled(self._reading)

    def set_busy(self, busy: bool):
        """等待 AI 翻譯時顯示旋轉圈；等待期間字幕不倒數消失。"""
        self.busy_ring.setVisible(busy)
        if busy:
            self._hide_timer.stop()
        elif (self.isVisible() and not self._hide_timer.isActive()
                and not self._reading):
            self._hide_timer.start(self._duration_ms())

    def dismiss(self):
        self.busy_ring.hide()
        if self._editing:
            # 直接關閉（✕）時放棄編輯，恢復原狀
            self._editing = False
            self.zh_edit.hide()
            self._apply_fonts()
            self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        self._hide_timer.stop()
        if self.isVisible():
            self._fade_out()

    # ---- 尺寸/字體 ----

    def _font_sizes(self):
        en = int(self.config.get(
            "subtitle", "font_size", default=DEFAULT_FONT_SIZE))
        return max(12, round(en * 0.72)), en

    def min_overlay_size(self) -> QSize:
        """最小尺寸隨字體連動：至少放得下中英各一行與邊距。"""
        zh, en = self._font_sizes()
        min_w = max(240, en * 6) + 28 + 16 + 102  # 102 = 控制欄（拖桿）寬
        min_h = 20 + 18 + int((zh + en) * 1.6) + 6
        if not self.replay_button.isHidden():
            # 控制欄放得下：⏸48 + 🔊36 + 兩組拖桿(22)＋標籤(14)
            min_h = max(min_h, 20 + 18 + 48 + 36 + (22 + 14) * 2 + 10)
        return QSize(min_w, min_h)

    def _apply_fonts(self, scale: float = 1.0):
        """用 QFont（而非 stylesheet）設定字級，heightForWidth 才算得準。
        字型/粗細/顏色依 config 的 subtitle 區段。"""
        zh, en = self._font_sizes()
        family = self.config.get("subtitle", "font_family", default="")
        weight = FONT_WEIGHTS.get(
            self.config.get("subtitle", "font_weight", default="bold"),
            QFont.Weight.Bold)
        zh_font = QFont(family or "Microsoft JhengHei")
        zh_font.setPixelSize(max(10, round(zh * scale)))
        zh_font.setWeight(weight)
        en_font = QFont(family or "Segoe UI")
        en_font.setWeight(weight)
        en_font.setPixelSize(max(12, round(en * scale)))
        self.zh_label.setFont(zh_font)

        color = QColor(self.config.get(
            "subtitle", "font_color", default=DEFAULT_FONT_COLOR))
        r, g, b = color.red(), color.green(), color.blue()
        # 中文行同色但稍微透明，維持主次層級
        self.zh_label.setStyleSheet(f"color: rgba({r},{g},{b},215);")
        # 單字方塊沿用英文行的字級與顏色
        for chip in self._chips:
            chip.apply_look(en_font, color)
        self.words_widget.updateGeometry()

    def refresh_style(self):
        """設定頁調整樣式時即時套用。"""
        if self.isVisible():
            self._relayout()
            self.update()

    def _set_lines(self, zh: str, en: str):
        self._set_words(en)
        self._apply_fonts()
        self.zh_label.setText(zh)
        self.zh_label.setVisible(bool(zh))

    def _reset_label_heights(self):
        for widget in (self.zh_label, self.words_widget):
            widget.setMinimumHeight(0)
            widget.setMaximumHeight(_MAX)

    def _relayout(self):
        """把文字塞進目前的框：照設定字體換行，放不下就逐步縮小字級。

        鎖住 label 高度讓排版的最小高度不超過視窗高度——這樣視窗
        不會為了顯示全部文字自己長高，完全尊重使用者設定的大小。"""
        if self._editing:
            return  # 編輯中 zh_label 是隱藏的，量出來會是 0
        # 用 isHidden()：第一次 _popup 還沒 show，isVisible() 會全是 False
        btn_w = (92 + 10) if not self.replay_button.isHidden() else 0
        avail_w = max(50, self.width() - 28 - 16 - btn_w)
        avail_h = max(20, self.height() - 20 - 18)
        has_words = bool(self._chips)
        both = self.zh_label.isVisible() and has_words
        spacing = 6 if both else 0
        zh_h = en_h = 0
        for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
            self._apply_fonts(scale)
            zh_h = (self.zh_label.heightForWidth(avail_w)
                    if self.zh_label.isVisible() else 0)
            # FlowLayout 沒有 QLabel 的 heightForWidth，改問排版本身
            en_h = self.words_widget.heightForWidth(avail_w) if has_words else 0
            if zh_h + en_h + spacing <= avail_h:
                break
        # 縮到底仍放不下時，各自裁切（不再撐大視窗）
        if self.zh_label.isVisible():
            self.zh_label.setFixedHeight(min(zh_h, avail_h))
        if has_words:
            self.words_widget.setFixedHeight(
                min(en_h, max(0, avail_h - zh_h - spacing))
                if both else min(en_h, avail_h))

    def _duration_ms(self) -> int:
        seconds = self.config.get(
            "subtitle", "duration_seconds", default=DEFAULT_DURATION_SECONDS)
        return max(3, int(seconds)) * 1000

    def apply_opacity(self):
        """設定頁調整時即時套用（顯示中且沒在淡入淡出才動）。"""
        if self.isVisible() and self._fade.state() != QPropertyAnimation.State.Running:
            self.setWindowOpacity(self.target_opacity())

    def _on_replay(self):
        # 重播期間字幕重新計時，聽的時候不會消失
        self._hide_timer.start(self._duration_ms())
        # 重播＝新的 submit → 暫停狀態作廢（圖示馬上復原成 ⏸）
        self._paused = False
        self._update_pause_button()
        self.replay_requested.emit()

    # ---- 中文行就地編輯 ----

    def begin_edit_empty(self):
        """全域 Enter 觸發：以空白輸入進入編輯（打新句子）。"""
        self._begin_edit(empty=True)

    def _begin_edit(self, empty: bool = False):
        if self._editing or not self.zh_label.isVisible():
            return
        self._editing = True
        self._hide_timer.stop()
        # 平常不搶焦點；編輯需要鍵盤 → 暫時拿掉 no-focus 旗標
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, False)
        self.show()
        self.zh_edit.setFont(self.zh_label.font())
        self.zh_edit.setPlainText("" if empty else self.zh_label.text())
        # 不能 hide() 中文行——排版會把英文行往上擠造成重疊；
        # 改成文字透明（占位不變），編輯框直接蓋在原位
        self.zh_label.setStyleSheet("color: transparent;")
        self.zh_edit.setGeometry(
            self.zh_label.geometry().adjusted(-4, -4, 4, 4))
        self.zh_edit.show()
        self.zh_edit.raise_()
        self.activateWindow()
        self.zh_edit.setFocus()
        cursor = self.zh_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.zh_edit.setTextCursor(cursor)

    def _finish_edit(self):
        if not self._editing:
            return
        self._editing = False
        import time
        self.last_edit_finished = time.monotonic()
        new_text = self.zh_edit.toPlainText().strip()
        old_text = self.zh_label.text()
        self.zh_edit.hide()
        self._apply_fonts()  # 恢復中文行的文字顏色
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        self.show()
        if new_text and new_text != old_text:
            self.zh_label.setText(new_text)
            self.set_busy(True)  # 等新結果回來（_popup 會收掉）
            self.retranslate_requested.emit(new_text)
        else:
            self._hide_timer.start(self._duration_ms())

    def _on_edit_focus_out(self, event):
        QTextEdit.focusOutEvent(self.zh_edit, event)
        # 點到編輯框外面 → 視同完成，重新翻譯
        self._finish_edit()

    # ---- 語速拖桿 ----

    def _sync_rate_from_config(self):
        rate = int(self.config.get("output", "tts_rate", default=200))
        self.rate_slider.blockSignals(True)
        self.rate_slider.setValue(rate)
        self.rate_slider.blockSignals(False)
        self.rate_label.setText(f"語速 {rate}")

    def _on_rate_changed(self, value: int):
        self.rate_label.setText(f"語速 {value}")
        # 拖動中不寫檔（每 pixel 存一次太重），放開時再存
        if not self.rate_slider.isSliderDown():
            self._save_rate(value)

    def _on_rate_drag_start(self):
        self._hide_timer.stop()  # 調整中不要倒數消失

    def _on_rate_drag_end(self):
        self._save_rate(self.rate_slider.value())
        self._hide_timer.start(self._duration_ms())

    def _save_rate(self, value: int):
        self.config.set("output", "tts_rate", int(value))
        self.rate_changed.emit(int(value))

    # ---- 音量拖桿 ----

    def _volume_fraction(self) -> float:
        volume = int(self.config.get("output", "tts_volume", default=100))
        return max(0, min(100, volume)) / 100.0

    def _sync_volume_from_config(self):
        volume = int(self.config.get("output", "tts_volume", default=100))
        self.volume_slider.blockSignals(True)
        self.volume_slider.setValue(volume)
        self.volume_slider.blockSignals(False)
        self.volume_label.setText(f"音量 {volume}%")

    def _on_volume_changed(self, value: int):
        self.volume_label.setText(f"音量 {value}%")
        # 拖動中不寫檔（每 pixel 存一次太重），放開時再存
        if not self.volume_slider.isSliderDown():
            self._save_volume(value)

    def _on_volume_drag_end(self):
        self._save_volume(self.volume_slider.value())
        self._hide_timer.start(self._duration_ms())

    def _save_volume(self, value: int):
        # 播放中不即時變音量（下一次播放/續播才生效）
        self.config.set("output", "tts_volume", int(value))
        self.volume_changed.emit(int(value))

    # ---- 顯示 ----

    def _popup(self, screen, duration_ms: int):
        self.busy_ring.hide()
        saved_pos = self.saved_pos()
        if saved_pos is not None:
            screen = QApplication.screenAt(saved_pos)
        screen = screen or QApplication.primaryScreen()
        geo = screen.availableGeometry()

        min_size = self.min_overlay_size()
        self.setMinimumSize(min_size)
        self.setMaximumSize(_MAX, _MAX)

        saved_size = self.saved_size()
        if saved_size is not None:
            saved_w, saved_h = saved_size
            width = max(int(saved_w), min_size.width())
            height = max(int(saved_h), min_size.height())
        else:
            width = max(min(int(geo.width() * 0.55), 860), min_size.width())
            # 依內容自動算高：暫時鎖寬讓 wordwrap 生效
            self._reset_label_heights()
            self.setFixedWidth(width)
            self.adjustSize()
            height = max(self.height(), min_size.height())
            self.setMinimumSize(min_size)
            self.setMaximumSize(_MAX, _MAX)
        self.resize(width, height)
        self._relayout()

        if saved_pos is not None:
            self.move(saved_pos)
        else:
            self.move(geo.center().x() - width // 2,
                      geo.bottom() - height - 72)

        self._hide_timer.stop()
        self._fade.stop()
        if not self.isVisible():
            self.setWindowOpacity(0.0)
            self.show()
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(self.target_opacity())
        self._fade.start()
        if not self._reading:  # 朗讀中不倒數，播完由 set_reading 重啟
            self._hide_timer.start(duration_ms)

    def _fade_out(self):
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
        bg = QColor(self.config.get(
            "subtitle", "bg_color", default=DEFAULT_BG_COLOR))
        bg.setAlpha(218)
        painter.setBrush(bg)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 14, 14)

    def resizeEvent(self, event):
        self.close_button.move(self.width() - self.close_button.width() - 6, 6)
        if self.isVisible():
            self._relayout()
        super().resizeEvent(event)

    # ---- 滑鼠互動：邊框調大小、內部拖曳移動、單擊關閉（通用行為見基底類別）----

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        # 開始操作（拖動或縮放）就先停止倒數，放開時由 _on_geometry_changed 重啟
        # 非左鍵（右鍵/中鍵）基底類別完全不處理，不能停倒數，否則永遠不會重啟
        if event.button() == Qt.MouseButton.LeftButton:
            self._hide_timer.stop()

    def _on_simple_click(self, pos):
        # 編輯中點卡片其他地方 → 視同完成編輯（重新翻譯）
        if self._editing:
            self._finish_edit()
            return
        # 點在原文行上 → 進入編輯；點其他地方 → 關閉
        if self.zh_label.isVisible() and self.zh_label.geometry().contains(pos):
            self._begin_edit()
        else:
            self.dismiss()

    def _on_geometry_changed(self):
        self.save_geometry()
        self._hide_timer.start(self._duration_ms())
