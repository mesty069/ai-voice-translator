import os
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

DEFAULT_RATE = 200  # pyttsx3/SAPI 預設語速（約每分鐘字數）

# 依目標語言挑選 SAPI 語音的關鍵字（比對 voice id/name 的大寫）
VOICE_HINTS = {
    "en": ("EN-US", "EN_US", "ENGLISH", "ZIRA", "DAVID", "MARK"),
    "zh": ("ZH-TW", "ZH_TW", "ZH-CN", "ZH_CN", "CHINESE", "HANHAN",
           "YATING", "HUIHUI", "KANGKANG"),
    "ja": ("JA-JP", "JA_JP", "JAPANESE", "HARUKA", "AYUMI", "ICHIRO"),
    "ko": ("KO-KR", "KO_KR", "KOREAN", "HEAMI"),
    "es": ("ES-ES", "ES-MX", "SPANISH", "HELENA", "SABINA"),
    "fr": ("FR-FR", "FRENCH", "HORTENSE", "JULIE"),
    "de": ("DE-DE", "GERMAN", "HEDDA", "STEFAN"),
    "vi": ("VI-VN", "VIETNAMESE", "AN"),
    "th": ("TH-TH", "THAI", "PATTARA"),
    "ru": ("RU-RU", "RUSSIAN", "IRINA", "PAVEL"),
}


def list_output_devices() -> list[str]:
    names = []
    try:
        default_hostapi = sd.query_hostapis(sd.default.hostapi)
    except Exception:
        default_hostapi = None
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] <= 0:
            continue
        if default_hostapi is not None and dev["hostapi"] != sd.default.hostapi:
            continue
        names.append(dev["name"])
    return names


def _resolve_device(name: str):
    if not name or name == "default":
        return None
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0 and dev["name"] == name:
            return idx
    return None


class _TTSWorker:
    """單一工作執行緒：所有合成（pyttsx3）與播放（sounddevice）只在
    這個執行緒發生——PortAudio 不是執行緒安全的，多執行緒同時
    stop/play/查詢串流會競態卡死。

    「信箱」只保留最新一筆請求：新請求會讓正在播的立刻中斷、
    排隊中未播的直接作廢——永不重疊、後來者覆蓋。

    暫停/繼續也走同一條路：GUI 執行緒只設旗標並 notify（不阻塞），
    真正的 sd.stop()/sd.play() 由 worker 在 _wait_playback 裡做。
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._pending = None
        self._thread = None
        self._wav_counter = 0
        # 以下狀態都由 _cond 保護
        self._playing = False       # 播放中（含暫停中）
        self._paused = False
        self._pause_requested = False
        self._resume_requested = False
        self._offset = 0            # 暫停時已播的樣本數
        self._volume = 1.0          # 目前播放用的音量（0.0–1.0）
        self._play_started = 0.0    # 本段開始播的 monotonic 時間

    def submit(self, text, device_name, rate, lang="en",
               on_start=None, on_done=None, volume=1.0):
        with self._cond:
            self._pending = (text, device_name, rate, lang,
                             on_start, on_done, volume)
            # 新請求一律清掉暫停狀態（暫停中按重播/點字都要能播）
            self._paused = False
            self._pause_requested = False
            self._resume_requested = False
            self._offset = 0
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, daemon=True, name="tts-worker")
                self._thread.start()
            self._cond.notify()

    # ---- 暫停／繼續（GUI 執行緒呼叫，不阻塞、不碰 sounddevice）----

    def pause(self) -> bool:
        """要求暫停；True 表示有東西可以暫停。"""
        with self._cond:
            if not self._playing or self._paused or self._pause_requested:
                return False
            self._pause_requested = True
            self._resume_requested = False
            self._cond.notify()
            return True

    def resume(self, volume=None) -> bool:
        """要求從暫停處繼續；volume 非 None 時同時換成新音量。"""
        with self._cond:
            if not self._paused or self._resume_requested:
                return False
            if volume is not None:
                self._volume = float(volume)
            self._resume_requested = True
            self._cond.notify()
            return True

    @property
    def is_playing(self) -> bool:
        with self._cond:
            return self._playing

    @property
    def is_paused(self) -> bool:
        with self._cond:
            return self._paused

    @property
    def played_offset(self) -> int:
        with self._cond:
            return self._offset

    def _has_newer(self) -> bool:
        with self._cond:
            return self._pending is not None

    def _run(self):
        try:
            import comtypes
            comtypes.CoInitialize()  # SAPI 需要 COM
        except Exception:
            pass
        while True:
            with self._cond:
                while self._pending is None:
                    self._cond.wait()
                (text, device_name, rate, lang,
                 on_start, on_done, volume) = self._pending
                self._pending = None
                self._volume = float(volume if volume is not None else 1.0)
                self._playing = False
                self._paused = False
                self._offset = 0
            try:
                sd.stop()  # 立刻切掉正在播的
            except Exception:
                pass
            if on_start is not None:
                try:
                    on_start()
                except Exception:
                    pass
            completed, error = False, None
            try:
                wav_path = self._synthesize(text, rate, lang)
                if not self._has_newer():
                    data, samplerate = sf.read(wav_path)
                    if not self._has_newer():
                        device = _resolve_device(device_name)
                        with self._cond:
                            self._playing = True
                        self._start_play(data, samplerate, device)
                        completed = self._wait_playback(
                            data, samplerate, device)
            except Exception as e:
                error = e
            with self._cond:
                self._playing = False
                self._paused = False
            if on_done is not None:
                try:
                    on_done(completed, error)
                except Exception:
                    pass

    def _synthesize(self, text, rate, lang="en") -> str:
        import pyttsx3
        self._wav_counter += 1
        wav_path = os.path.join(
            tempfile.gettempdir(),
            f"ai_translator_tts_{self._wav_counter % 8}.wav")
        engine = pyttsx3.init()
        engine.setProperty("rate", int(rate))
        hints = VOICE_HINTS.get(lang, VOICE_HINTS["en"])
        for voice in engine.getProperty("voices"):
            haystack = (voice.id + " " + (voice.name or "")).upper()
            if any(h in haystack for h in hints):
                engine.setProperty("voice", voice.id)
                break
        # 找不到對應語言的語音時用系統預設（發音可能不道地，
        # 需在 Windows 設定安裝該語言的語音套件）
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        return wav_path

    @staticmethod
    def _apply_volume(data, volume: float):
        """音量縮放：float32、clip 到 [-1,1]（音量 1.0 原樣不動）。"""
        if abs(float(volume) - 1.0) < 1e-6:
            return data
        scaled = np.asarray(data, dtype="float32") * float(volume)
        return np.clip(scaled, -1.0, 1.0)

    def _start_play(self, data, samplerate, device):
        """只在 worker 執行緒呼叫。"""
        with self._cond:
            volume, offset = self._volume, self._offset
            self._play_started = time.monotonic()
        sd.play(self._apply_volume(data[offset:], volume), samplerate,
                device=device)

    def _wait_playback(self, data, samplerate, device) -> bool:
        """等播放結束（True）或被新請求打斷（False）。

        暫停時停在這裡等 resume 或新請求——既不算播完也不算被打斷，
        所以 on_done 不會在暫停時觸發。"""
        while True:
            with self._cond:
                self._cond.wait(0.05)   # 有新請求/暫停/繼續會立刻被叫醒
                has_newer = self._pending is not None
                pause_requested = self._pause_requested
                resume_requested = self._resume_requested
                paused = self._paused
            if has_newer:
                try:
                    sd.stop()
                except Exception:
                    pass
                return False
            if pause_requested:
                with self._cond:
                    elapsed = max(0.0, time.monotonic() - self._play_started)
                    self._offset = min(
                        len(data), self._offset + int(elapsed * samplerate))
                    self._pause_requested = False
                    self._paused = True
                try:
                    sd.stop()
                except Exception:
                    pass
                continue
            if paused:
                if resume_requested:
                    with self._cond:
                        self._resume_requested = False
                        self._paused = False
                    self._start_play(data, samplerate, device)
                continue
            try:
                if not sd.get_stream().active:
                    return True
            except Exception:
                return True


_worker = _TTSWorker()


def submit(text: str, device_name: str = "default", rate: int = DEFAULT_RATE,
           lang: str = "en", on_start=None, on_done=None,
           volume: float = 1.0):
    """非阻塞：把朗讀請求丟給 TTS 工作執行緒（最新的請求覆蓋舊的）。

    lang：語言代碼，用來挑選對應的語音（挑不到用系統預設）。
    volume：0.0–1.0 的音量倍率（播放前縮放並 clip）。
    on_start()：開始處理（合成前）時回呼。
    on_done(completed, error)：結束時回呼——completed=True 表示完整
    播完；False 表示被更新的請求取代或發生錯誤（error 非 None）。
    暫停不會觸發 on_done（要等 resume 播完或被新請求取代）。
    回呼都在 TTS 執行緒上執行，別做重活。
    """
    _worker.submit(text, device_name, rate, lang, on_start, on_done, volume)


def pause() -> bool:
    """暫停目前的朗讀（非阻塞）；True 表示真的有東西被暫停。"""
    return _worker.pause()


def resume(volume: float | None = None) -> bool:
    """從暫停處繼續朗讀（非阻塞）；True 表示真的有東西被續播。"""
    return _worker.resume(volume)


def playback_state() -> dict:
    """目前播放狀態：{"playing": 播放中（含暫停）, "paused": 暫停中}。"""
    return {"playing": _worker.is_playing, "paused": _worker.is_paused}
