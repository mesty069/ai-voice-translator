import queue
import threading
import time

from PySide6.QtCore import QObject, Signal

from .local_translate import LocalTranslator
from .system_audio import SystemAudioCapture


class SystemCaptionsController(QObject):
    """把「系統聲音 → 文字 → 母語翻譯」串起來。

    擷取與處理都在自己的執行緒，不佔用麥克風流程的 executor。
    麥克風正在使用時先讓路，避免使用者說話被系統字幕的辨識卡住。
    """

    caption_ready = Signal(str, str)      # 原文, 母語翻譯
    state_changed = Signal(str, str)      # state, message
    error_occurred = Signal(str)

    MAX_QUEUE = 3  # 積壓超過就丟掉最舊的，即時性優先

    def __init__(self, config, stt, mic_busy, parent=None):
        super().__init__(parent)
        self.config = config
        self.stt = stt
        self._mic_busy = mic_busy
        self._mic_wait_timeout = 10.0
        self._translator = LocalTranslator(
            engine=config.get("system_captions", "engine",
                              default="nllb-600m"),
            compute_device=config.get("system_captions", "compute_device",
                                      default="auto"))
        self._capture = None
        self._capture_factory = SystemAudioCapture
        self._queue = queue.Queue()
        self._worker = None
        self._running = False
        self._generation = 0   # 每次 start() 遞增；舊 worker 看到不一致就退出、不發訊號

    # ---- 生命週期 ----

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._running = True
        self._generation += 1
        gen = self._generation
        # 換新佇列：舊 worker 就算晚醒來也吃不到新的分段
        self._queue = queue.Queue()
        self._translator.set_engine(
            self.config.get("system_captions", "engine", default="nllb-600m"))
        self._translator.set_compute_device(
            self.config.get("system_captions", "compute_device",
                            default="auto"))
        self._worker = threading.Thread(
            target=self._work, args=(gen,), daemon=True, name="system-captions")
        self._worker.start()
        self._capture = self._capture_factory(
            device_name=self.config.get("system_captions", "device",
                                        default="default"),
            on_segment=lambda audio, g=gen: self._on_segment(audio, g),
            on_error=lambda error, g=gen: self._on_capture_error(error, g),
            silence_ms=self.config.get("system_captions", "segment_silence_ms",
                                       default=600),
            max_seconds=self.config.get("system_captions", "max_segment_sec",
                                        default=8))
        self._capture.start()
        self.state_changed.emit("listening", "正在聽系統聲音…")

    def stop(self):
        # 沒在跑的話直接返回，避免 UI 重複呼叫 stop 時狂洗狀態列
        if not self._running and self._capture is None:
            return
        self._running = False
        # 世代先變號，就算舊 worker 之後才醒來，也無法再發訊號或吃到新佇列
        self._generation += 1
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        self._queue.put(None)  # 叫醒目前這個佇列的工作執行緒
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(timeout=2.0)
        self.state_changed.emit("idle", "已停止系統聲音字幕")

    # ---- 語言 ----

    def _languages(self):
        """回傳 (辨識語言, 翻譯目標語言)。"""
        native = self.config.get("language", "source", default="zh")
        target = self.config.get("language", "target", default="en")
        spoken = self.config.get("system_captions", "language", default="")
        return (spoken or target), native

    # ---- 管線 ----

    def _on_segment(self, audio, gen):
        if gen != self._generation or not self._running:
            return  # 舊世代的擷取執行緒，不得餵進新佇列
        # 先取本地變數，避免 start() 同時換新佇列造成 trim/put 對不上
        q = self._queue
        while q.qsize() >= self.MAX_QUEUE:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        q.put(audio)

    def _on_capture_error(self, error, gen):
        if gen != self._generation:
            return  # 舊世代擷取執行緒遲來的錯誤，忽略即可
        self._running = False
        self.error_occurred.emit(f"系統聲音擷取失敗：{error}")
        self.state_changed.emit("error", "系統聲音擷取失敗")
        self._queue.put(None)  # 叫醒卡在 queue.get() 的工作執行緒，避免永遠卡住

    def _work(self, gen):
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
        while self._running and gen == self._generation:
            audio = self._queue.get()
            if gen != self._generation or not self._running:
                break
            if audio is None:
                break
            try:
                self._process(audio, gen)
            except Exception as e:
                self.error_occurred.emit(f"系統字幕處理失敗：{e}")

    def _process(self, audio, gen):
        # 麥克風優先：使用者正在說話時先讓路
        deadline = time.monotonic() + self._mic_wait_timeout
        while self._mic_busy() and time.monotonic() < deadline:
            time.sleep(0.05)

        spoken, native = self._languages()
        text = (self.stt.transcribe(audio, language=spoken) or "").strip()
        if gen != self._generation or not self._running:
            return  # stop() 期間或已重啟，舊結果不發訊號
        if not text:
            return
        translated = self._translator.translate(text, spoken, native)
        if gen != self._generation or not self._running:
            return  # translate() 耗時，回來後可能已經被 stop()/重啟
        self.caption_ready.emit(text, translated)
