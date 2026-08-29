import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PySide6.QtCore import QObject, Signal

from .ai.base import TranslationError
from .ai.factory import create_provider
from .config import Config, lang_prompt_name
from .core import tts
from .core.hotkey import HotkeyListener
from .core.mic_isolation import MicIsolation
from .core.recorder import Recorder, SAMPLE_RATE
from .core.stt import SpeechToText


def _co_initialize():
    """executor 執行緒要跑 SAPI TTS（COM），先初始化 apartment。"""
    import comtypes
    comtypes.CoInitialize()

MIN_DURATION_SECONDS = 0.35


def _is_model_cached(model_size: str) -> bool:
    import os
    from pathlib import Path
    hub = Path(os.environ.get(
        "HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return hub.is_dir() and any(
        hub.glob(f"models--*faster-whisper-{model_size}"))


class AppController(QObject):
    """組裝熱鍵→錄音→禁音→STT→AI 的核心流程，用 signal 回報 UI。"""

    state_changed = Signal(str, str)          # state: loading/idle/recording/processing/error, message
    result_ready = Signal(str, str, str)      # raw, refined, english
    error_occurred = Signal(str)
    copy_requested = Signal(str)              # 需在 GUI 執行緒寫剪貼簿
    level_changed = Signal(float)             # 錄音中每 50ms 回報一次峰值 0.0~1.0
    replay_hotkey_pressed = Signal()          # 朗讀熱鍵被按下（由 UI 決定是否有效）
    enter_pressed = Signal()                  # 全域 Enter（字幕顯示時進入編輯）
    tts_playing = Signal(bool)                # 朗讀開始/結束（字幕倒數依此暫停）
    wait_hint_visible = Signal(bool)          # 隔離空窗期（畫面中央「請稍等再說」）

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.recorder = Recorder(
            config.get("recording", "device", default="default"),
            level_callback=self._on_level)
        self._last_level_emit = 0.0
        self.mic_isolation = MicIsolation()
        self._isolated = False
        self._last_english = ""
        self._session_peak = 0.0
        self._mic_warned = False
        self._record_started_at = 0.0
        self.stt = SpeechToText(
            model_size=config.get("stt", "model_size", default="small"),
            device=config.get("stt", "device", default="cpu"),
            compute_type=config.get("stt", "compute_type", default="int8"),
        )
        self.hotkey = HotkeyListener(self._on_hotkey_press, self._on_hotkey_release)
        self.replay_hotkey = HotkeyListener(
            self.replay_hotkey_pressed.emit, lambda: None)
        self.grammar_hotkey = HotkeyListener(
            self._on_grammar_press, self._on_hotkey_release)
        self._mode = "translate"  # translate | grammar
        self.enter_listener = HotkeyListener(
            self.enter_pressed.emit, lambda: None)
        # 單執行緒 executor：天然序列化「開始錄音 → 停止並處理」。
        self._executor = ThreadPoolExecutor(
            max_workers=1, initializer=_co_initialize)
        self._session_active = False
        self._model_ready = threading.Event()  # 載入完成（成功或失敗）時 set
        self._tts_token = 0                    # 最新一次朗讀請求的編號
        self._tts_token_lock = threading.Lock()

    # ---- 生命週期 ----

    def start(self):
        self.apply_hotkey()
        self.apply_replay_hotkey()
        self.apply_grammar_hotkey()
        try:
            self.enter_listener.configure("keyboard", "enter")
        except ValueError:
            pass
        threading.Thread(target=self._load_model, daemon=True).start()

    def shutdown(self):
        self.hotkey.stop()
        self.replay_hotkey.stop()
        self.grammar_hotkey.stop()
        self.enter_listener.stop()
        self._executor.shutdown(wait=False)

    def apply_grammar_hotkey(self):
        enabled = self.config.get("grammar", "enabled", default=False)
        hotkey_type = self.config.get("grammar", "hotkey_type", default="keyboard")
        key_name = self.config.get("grammar", "hotkey_key", default="f10")
        if not enabled or not key_name:
            self.grammar_hotkey.stop()
            return
        try:
            self.grammar_hotkey.configure(hotkey_type, key_name)
        except ValueError as e:
            self.error_occurred.emit(str(e))

    def apply_replay_hotkey(self):
        hotkey_type = self.config.get("replay_hotkey", "type", default="keyboard")
        key_name = self.config.get("replay_hotkey", "key", default="")
        if not key_name:
            self.replay_hotkey.stop()
            return
        try:
            self.replay_hotkey.configure(hotkey_type, key_name)
        except ValueError as e:
            self.error_occurred.emit(str(e))

    def _load_model(self):
        size = self.stt.model_size
        if _is_model_cached(size):
            self.state_changed.emit("loading", f"正在載入語音模型（{size}）…")
        else:
            self.state_changed.emit(
                "loading", f"首次使用 {size} 模型，正在下載（之後會存在本機，不再重下）…")
        try:
            self.stt.load()
        except Exception as e:
            self.state_changed.emit("error", f"語音模型載入失敗：{e}")
            return
        finally:
            self._model_ready.set()  # 失敗也要 set，喚醒等待中的錄音處理
        if not self._session_active:  # 錄音/處理中就別蓋掉狀態列
            self.state_changed.emit("idle", "就緒，按住熱鍵開始說話")

    def reload_model(self, model_size: str):
        def _reload():
            self._model_ready.clear()
            self.state_changed.emit("loading", f"正在切換語音模型（{model_size}）…")
            try:
                self.stt.reload(model_size)
            except Exception as e:
                self.state_changed.emit("error", f"語音模型載入失敗：{e}")
                return
            finally:
                self._model_ready.set()
            self.state_changed.emit("idle", "就緒，按住熱鍵開始說話")
        threading.Thread(target=_reload, daemon=True).start()

    def apply_recording_device(self):
        self.recorder.set_device(
            self.config.get("recording", "device", default="default"))

    def apply_hotkey(self):
        hotkey_type = self.config.get("hotkey", "type", default="keyboard")
        key_name = self.config.get("hotkey", "key", default="f9")
        try:
            self.hotkey.configure(hotkey_type, key_name)
        except ValueError as e:
            self.error_occurred.emit(str(e))

    # ---- 文字輸入翻譯（GUI 執行緒呼叫）----

    def translate_text(self, text: str):
        text = text.strip()
        if not text:
            return
        if self._session_active:
            self.state_changed.emit("processing", "還在處理上一句，請稍候…")
            return
        self._session_active = True
        self._executor.submit(self._process_text, text)

    def _langs(self):
        """回傳 (母語代碼, 目標語言代碼)。"""
        return (self.config.get("language", "source", default="zh"),
                self.config.get("language", "target", default="en"))

    def _process_text(self, text: str):
        try:
            src, tgt = self._langs()
            self.state_changed.emit("processing", "AI 梳理與翻譯中…")
            try:
                provider = create_provider(self.config.get("ai", default={}))
                result = provider.refine_and_translate(
                    text, lang_prompt_name(src), lang_prompt_name(tgt))
            except TranslationError as e:
                self.result_ready.emit(text, "", "")
                self.state_changed.emit("error", f"翻譯失敗：{e}")
                return
            self.result_ready.emit(text, result.refined, result.english)
            self.state_changed.emit("idle", "完成")
            self._handle_outputs(result.english)
        finally:
            self._session_active = False

    # ---- 熱鍵回呼（pynput 執行緒）----

    def _on_hotkey_press(self):
        if self._session_active:
            return
        # 模型載入中也照樣錄音——錄音不需要模型，放開後等模型就緒再辨識
        self._mode = "translate"
        self._session_active = True
        self._executor.submit(self._start_recording)

    def _on_grammar_press(self):
        if self._session_active:
            return
        self._mode = "grammar"
        self._session_active = True
        self._executor.submit(self._start_recording)

    def _on_hotkey_release(self):
        if not self._session_active:
            return
        self._executor.submit(self._stop_and_process)

    # ---- 在 executor 執行緒 ----

    def _on_level(self, peak: float):
        # PortAudio 執行緒呼叫，節流到每 50ms 一次再丟給 UI
        now = time.monotonic()
        self._session_peak = max(self._session_peak, peak)
        # 錄了超過 1.2 秒還是數位全零 → 立刻主動提示（每次錄音只提示一次）
        if (not self._mic_warned
                and now - self._record_started_at > 1.2
                and self._session_peak < 0.0005):
            self._mic_warned = True
            self.error_occurred.emit(
                "偵測不到麥克風聲音——請檢查麥克風本體的實體靜音鍵（LED 燈），"
                "或到設定頁確認選了正確的錄音裝置")
        if now - self._last_level_emit >= 0.05:
            self._last_level_emit = now
            self.level_changed.emit(peak)

    def _start_recording(self):
        hinted = False
        try:
            self._session_peak = 0.0
            self._mic_warned = False
            self._record_started_at = time.monotonic()
            isolated = 0
            if self.config.get("recording", "isolate_other_devices",
                               default=True):
                self.state_changed.emit("processing", "正在隔離麥克風…")
                # 這段空窗期講話會漏音 → 畫面中央顯示「請稍等再說」
                self.wait_hint_visible.emit(True)
                hinted = True
                try:
                    isolated = self.mic_isolation.isolate(
                        self.recorder.device_name)
                    self._isolated = True
                except Exception as e:
                    self.error_occurred.emit(
                        f"隔離麥克風失敗（繼續錄音）：{e}")
                if isolated:
                    # Discord/Teams 跟隨預設裝置切換需要時間，先等它們
                    # 切完再開錄、再亮紅色——紅色代表「已隔離，可以講了」
                    settle_ms = int(self.config.get(
                        "recording", "isolation_settle_ms", default=700))
                    time.sleep(max(0, settle_ms) / 1000)
            self.recorder.start()
            prefix = ("文法檢查，錄音中…" if self._mode == "grammar"
                      else "錄音中…")
            self.state_changed.emit(
                "recording",
                prefix + ("（麥克風已隔離，其他軟體聽不到）" if isolated else ""))
        except Exception as e:
            self._restore_isolation()
            self._session_active = False
            self.state_changed.emit("error", f"無法開始錄音：{e}")
        finally:
            # 可以開口了（或出錯了）→ 一定要收掉提示
            if hinted:
                self.wait_hint_visible.emit(False)

    def _restore_isolation(self):
        if not self._isolated:
            return
        try:
            self.mic_isolation.restore()
        except Exception as e:
            self.error_occurred.emit(f"恢復其他錄音裝置失敗：{e}")
        finally:
            self._isolated = False

    def _stop_and_process(self):
        # 整個流程包在 try/finally：任何一步出錯都不能讓 _session_active
        # 卡在 True（否則熱鍵從此沒反應）
        try:
            try:
                audio = self.recorder.stop()
            finally:
                self._restore_isolation()
            duration = len(audio) / SAMPLE_RATE
            if duration < MIN_DURATION_SECONDS:
                self.state_changed.emit("idle", "按住時間太短，已忽略")
                return
            if not self.stt.is_ready:
                self.state_changed.emit(
                    "processing", "錄音已保存，等語音模型載入完成後立即翻譯…")
                self._model_ready.wait(timeout=300)
                if not self.stt.is_ready:
                    self.state_changed.emit(
                        "error", "語音模型尚未就緒（載入失敗或逾時），這段錄音無法辨識")
                    return
            grammar_mode = self._mode == "grammar"
            src, tgt = self._langs()
            self.state_changed.emit("processing", "語音辨識中…")
            try:
                raw_text = self.stt.transcribe(
                    audio, language=tgt if grammar_mode else src)
            except Exception as e:
                self.state_changed.emit("error", f"語音辨識失敗：{e}")
                return
            if not raw_text:
                peak = float(np.abs(audio).max()) if len(audio) else 0.0
                if peak < 0.0005:
                    self.state_changed.emit(
                        "error",
                        "麥克風輸出是數位全零——通常是麥克風本體的實體靜音鍵"
                        "被按到了（檢查機身 LED），或裝置被拔除")
                elif peak < 0.02:
                    self.state_changed.emit(
                        "idle",
                        f"沒有辨識到內容，錄到的音量很小（峰值 {peak:.3f}）"
                        "——請確認設定頁選的麥克風正確、離麥克風近一點")
                else:
                    self.state_changed.emit("idle", "沒有辨識到任何內容，請再試一次")
                return
            if grammar_mode:
                self.state_changed.emit("processing", "AI 檢查文法中…")
                try:
                    provider = create_provider(self.config.get("ai", default={}))
                    check = provider.check_grammar(
                        raw_text, lang_prompt_name(tgt))
                except TranslationError as e:
                    self.state_changed.emit("error", f"文法檢查失敗：{e}")
                    return
                if not check.has_errors:
                    # 沒錯誤 → 不回覆（只在狀態列輕聲告知）
                    self.state_changed.emit("idle", f"✔ 文法正確：{raw_text}")
                    return
                self.result_ready.emit(raw_text, f"✗ {raw_text}", check.corrected)
                self.state_changed.emit("idle", "已修正文法")
                self._handle_outputs(check.corrected)
                return
            self.state_changed.emit("processing", "AI 梳理與翻譯中…")
            try:
                provider = create_provider(self.config.get("ai", default={}))
                result = provider.refine_and_translate(
                    raw_text, lang_prompt_name(src), lang_prompt_name(tgt))
            except TranslationError as e:
                self.result_ready.emit(raw_text, "", "")
                self.state_changed.emit("error", f"翻譯失敗：{e}")
                return
            self.result_ready.emit(raw_text, result.refined, result.english)
            self.state_changed.emit("idle", "完成")
            self._handle_outputs(result.english)
        except Exception as e:
            self.state_changed.emit("error", f"處理錄音時發生錯誤：{e}")
        finally:
            self._session_active = False

    def _handle_outputs(self, english: str):
        self._last_english = english
        if self.config.get("output", "auto_copy", default=False):
            self.copy_requested.emit(english)
        if self.config.get("output", "tts_enabled", default=False):
            self._speak_async(english)

    def _speak_async(self, english: str):
        """把朗讀交給 TTS 專用工作執行緒（tts.submit），最新請求覆蓋舊的。
        不佔用 executor，也不會有多執行緒同時碰音訊裝置的競態。"""
        with self._tts_token_lock:
            self._tts_token += 1
            token = self._tts_token

        def on_start():
            self.tts_playing.emit(True)

        def on_done(completed, error):
            if error is not None:
                self.error_occurred.emit(f"語音播放失敗：{error}")
            # 只有「最新」的請求負責宣告結束；被取代的舊請求不出聲，
            # 避免舊的結束訊號把新的播放中狀態蓋掉
            with self._tts_token_lock:
                is_latest = token == self._tts_token
            if is_latest:
                self.tts_playing.emit(False)

        tts.submit(
            english,
            self.config.get("output", "tts_device", default="default"),
            self.config.get("output", "tts_rate", default=tts.DEFAULT_RATE),
            self.config.get("language", "target", default="en"),
            on_start, on_done)

    def replay_tts(self):
        """重新播放最後一句英文語音（GUI 執行緒呼叫，不論 TTS 開關）。"""
        if self._last_english:
            self._speak_async(self._last_english)

    def speak_text(self, text: str):
        """朗讀任意英文文字（主畫面的 🔊 按鈕，不論 TTS 開關）。"""
        text = (text or "").strip()
        if text:
            self._speak_async(text)
