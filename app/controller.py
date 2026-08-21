import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, Signal

from .ai.base import TranslationError
from .ai.factory import create_provider
from .config import Config
from .core import tts
from .core.hotkey import HotkeyListener
from .core.mic_guard import MicGuard
from .core.recorder import Recorder, SAMPLE_RATE
from .core.stt import SpeechToText

MIN_DURATION_SECONDS = 0.35


class AppController(QObject):
    """組裝熱鍵→錄音→禁音→STT→AI 的核心流程，用 signal 回報 UI。"""

    state_changed = Signal(str, str)          # state: loading/idle/recording/processing/error, message
    result_ready = Signal(str, str, str)      # raw, refined, english
    error_occurred = Signal(str)
    copy_requested = Signal(str)              # 需在 GUI 執行緒寫剪貼簿

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.recorder = Recorder()
        self.mic_guard = MicGuard()
        self.stt = SpeechToText(
            model_size=config.get("stt", "model_size", default="small"),
            device=config.get("stt", "device", default="cpu"),
            compute_type=config.get("stt", "compute_type", default="int8"),
        )
        self.hotkey = HotkeyListener(self._on_hotkey_press, self._on_hotkey_release)
        # 單執行緒 executor：mute/unmute 必須在同一 COM apartment，
        # 同時天然序列化「開始錄音 → 停止並處理」。
        self._executor = ThreadPoolExecutor(
            max_workers=1, initializer=MicGuard.co_initialize)
        self._session_active = False
        self._muted = False

    # ---- 生命週期 ----

    def start(self):
        self.apply_hotkey()
        threading.Thread(target=self._load_model, daemon=True).start()

    def shutdown(self):
        self.hotkey.stop()
        self._executor.shutdown(wait=False)

    def _load_model(self):
        size = self.stt.model_size
        self.state_changed.emit(
            "loading", f"正在載入語音模型（{size}），首次執行會自動下載…")
        try:
            self.stt.load()
        except Exception as e:
            self.state_changed.emit("error", f"語音模型載入失敗：{e}")
            return
        self.state_changed.emit("idle", "就緒，按住熱鍵開始說話")

    def reload_model(self, model_size: str):
        def _reload():
            self.state_changed.emit("loading", f"正在切換語音模型（{model_size}）…")
            try:
                self.stt.reload(model_size)
            except Exception as e:
                self.state_changed.emit("error", f"語音模型載入失敗：{e}")
                return
            self.state_changed.emit("idle", "就緒，按住熱鍵開始說話")
        threading.Thread(target=_reload, daemon=True).start()

    def apply_hotkey(self):
        hotkey_type = self.config.get("hotkey", "type", default="keyboard")
        key_name = self.config.get("hotkey", "key", default="f9")
        try:
            self.hotkey.configure(hotkey_type, key_name)
        except ValueError as e:
            self.error_occurred.emit(str(e))

    # ---- 熱鍵回呼（pynput 執行緒）----

    def _on_hotkey_press(self):
        if self._session_active:
            return
        if not self.stt.is_ready:
            self.state_changed.emit("loading", "語音模型還在載入，請稍候…")
            return
        self._session_active = True
        self._executor.submit(self._start_recording)

    def _on_hotkey_release(self):
        if not self._session_active:
            return
        self._executor.submit(self._stop_and_process)

    # ---- 在 executor 執行緒 ----

    def _start_recording(self):
        try:
            if self.config.get("mute_other_apps", default=True):
                try:
                    self.mic_guard.mute_others()
                    self._muted = True
                except Exception as e:
                    self.error_occurred.emit(f"靜音其他程式失敗（繼續錄音）：{e}")
            self.recorder.start()
            self.state_changed.emit("recording", "錄音中…（其他程式麥克風已靜音）"
                                    if self._muted else "錄音中…")
        except Exception as e:
            self._restore_mic()
            self._session_active = False
            self.state_changed.emit("error", f"無法開始錄音：{e}")

    def _stop_and_process(self):
        try:
            audio = self.recorder.stop()
        finally:
            self._restore_mic()
        try:
            duration = len(audio) / SAMPLE_RATE
            if duration < MIN_DURATION_SECONDS:
                self.state_changed.emit("idle", "按住時間太短，已忽略")
                return
            self.state_changed.emit("processing", "語音辨識中…")
            try:
                raw_text = self.stt.transcribe(audio)
            except Exception as e:
                self.state_changed.emit("error", f"語音辨識失敗：{e}")
                return
            if not raw_text:
                self.state_changed.emit("idle", "沒有辨識到任何內容，請再試一次")
                return
            self.state_changed.emit("processing", "AI 梳理與翻譯中…")
            try:
                provider = create_provider(self.config.get("ai", default={}))
                result = provider.refine_and_translate(raw_text)
            except TranslationError as e:
                self.result_ready.emit(raw_text, "", "")
                self.state_changed.emit("error", f"翻譯失敗：{e}")
                return
            self.result_ready.emit(raw_text, result.refined, result.english)
            self.state_changed.emit("idle", "完成")
            self._handle_outputs(result.english)
        finally:
            self._session_active = False

    def _restore_mic(self):
        if not self._muted:
            return
        try:
            self.mic_guard.unmute_others()
        except Exception as e:
            self.error_occurred.emit(f"恢復其他程式麥克風失敗：{e}")
        finally:
            self._muted = False

    def _handle_outputs(self, english: str):
        if self.config.get("output", "auto_copy", default=False):
            self.copy_requested.emit(english)
        if self.config.get("output", "tts_enabled", default=False):
            try:
                tts.speak(english, self.config.get(
                    "output", "tts_device", default="default"))
            except Exception as e:
                self.error_occurred.emit(f"語音播放失敗：{e}")
