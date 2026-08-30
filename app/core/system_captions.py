from PySide6.QtCore import QObject, Signal

from .local_translate import LocalTranslator
from .streaming_captions import DISPLAY_ROWS, StreamingCaptionEngine
from .system_audio import RollingAudioBuffer, SystemAudioCapture


class SystemCaptionsController(QObject):
    """組裝「系統聲音擷取 → 滾動緩衝 → 串流引擎 → 本機翻譯」，把回呼轉成 Qt 訊號。

    世代編號：每次 start() 遞增；stop() 之後舊 capture／engine 的回呼一律忽略。
    """

    rows_changed = Signal(object)          # list[Row]
    sentence_finalized = Signal(str, str)  # 完成句 (原文, 翻譯) → 歷史面板
    state_changed = Signal(str, str)       # state, message
    fatal_error = Signal(str)              # 擷取死掉、模型載不起來 → 關閉功能

    def __init__(self, config, stt, mic_busy, parent=None):
        super().__init__(parent)
        self.config = config
        self.stt = stt
        self._mic_busy = mic_busy
        self._translator = LocalTranslator(
            engine=config.get("system_captions", "engine", default="nllb-600m"),
            compute_device=config.get("system_captions", "compute_device",
                                      default="auto"))
        self._capture_factory = SystemAudioCapture
        self._engine_factory = StreamingCaptionEngine
        self._capture = None
        self._engine = None
        self._buffer = None
        self._running = False
        self._generation = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._running = True
        self._generation += 1
        gen = self._generation
        self._translator.set_engine(
            self.config.get("system_captions", "engine", default="nllb-600m"))
        self._translator.set_compute_device(
            self.config.get("system_captions", "compute_device", default="auto"))
        self._buffer = RollingAudioBuffer()
        buffer = self._buffer
        self._engine = self._engine_factory(
            buffer, self.stt, self._translator,
            languages=self._languages, mic_busy=self._mic_busy,
            on_rows=lambda rows, g=gen: self._guarded(g, self.rows_changed.emit, rows),
            on_state=lambda s, m, g=gen: self._guarded(g, self.state_changed.emit, s, m),
            on_fatal=lambda msg, g=gen: self._on_fatal(msg, g),
            on_final=lambda o, t, g=gen: self._guarded(g, self.sentence_finalized.emit, o, t),
            display_rows=self.config.get("system_captions", "display_rows",
                                         default=DISPLAY_ROWS))
        self._engine.start()
        self._capture = self._capture_factory(
            device_name=self.config.get("system_captions", "device",
                                        default="default"),
            on_frames=lambda frames, g=gen, b=buffer: self._on_frames(frames, g, b),
            on_error=lambda error, g=gen: self._on_fatal(
                f"系統聲音擷取失敗：{error}", g))
        self._capture.start()
        self.state_changed.emit("listening", "正在聽系統聲音…")

    def stop(self):
        if not self._running and self._capture is None and self._engine is None:
            return
        self._teardown()
        self.state_changed.emit("idle", "已停止系統聲音字幕")

    def set_display_rows(self, n: int):
        if self._engine is not None:
            self._engine.set_display_rows(n)

    # ---- 內部 ----

    def _languages(self):
        """(辨識語言, 翻譯目標語言)：預設辨識「目標語言」、翻成「母語」。"""
        native = self.config.get("language", "source", default="zh")
        target = self.config.get("language", "target", default="en")
        spoken = self.config.get("system_captions", "language", default="")
        return (spoken or target), native

    def _teardown(self):
        """停掉擷取與引擎並讓舊回呼失效（stop() 與 _on_fatal 共用）。

        兩條執行緒是綁在一起的：擷取死了引擎只會空轉 poll，引擎死了擷取
        還在錄 loopback，所以任一邊致命都要把另一邊也收掉。
        """
        self._running = False
        self._generation += 1   # 先變號：舊回呼從此無效
        capture, self._capture = self._capture, None
        engine, self._engine = self._engine, None
        if capture is not None:
            capture.stop()      # join ≤ 0.2s
        if engine is not None:
            engine.stop()       # join ≤ 0.1s

    def _guarded(self, gen, emit, *args):
        if gen == self._generation and self._running:
            emit(*args)

    def _on_frames(self, frames, gen, buffer):
        if gen == self._generation and self._running:
            buffer.append(frames)

    def _on_fatal(self, message, gen):
        if gen != self._generation:
            return          # 世代已變（stop 過或另一邊先致命）→ 只提示一次
        self._teardown()    # 不發 "idle"：狀態要停在 error
        self.fatal_error.emit(message)
        self.state_changed.emit("error", message)
