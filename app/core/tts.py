import os
import tempfile
import threading
import time

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
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._pending = None
        self._thread = None
        self._wav_counter = 0

    def submit(self, text, device_name, rate, lang="en",
               on_start=None, on_done=None):
        with self._cond:
            self._pending = (text, device_name, rate, lang, on_start, on_done)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, daemon=True, name="tts-worker")
                self._thread.start()
            self._cond.notify()

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
                 on_start, on_done) = self._pending
                self._pending = None
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
                        sd.play(data, samplerate,
                                device=_resolve_device(device_name))
                        completed = self._wait_playback()
            except Exception as e:
                error = e
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

    def _wait_playback(self) -> bool:
        """等播放結束（True）或被新請求打斷（False）。"""
        while True:
            time.sleep(0.05)
            if self._has_newer():
                try:
                    sd.stop()
                except Exception:
                    pass
                return False
            try:
                if not sd.get_stream().active:
                    return True
            except Exception:
                return True


_worker = _TTSWorker()


def submit(text: str, device_name: str = "default", rate: int = DEFAULT_RATE,
           lang: str = "en", on_start=None, on_done=None):
    """非阻塞：把朗讀請求丟給 TTS 工作執行緒（最新的請求覆蓋舊的）。

    lang：語言代碼，用來挑選對應的語音（挑不到用系統預設）。
    on_start()：開始處理（合成前）時回呼。
    on_done(completed, error)：結束時回呼——completed=True 表示完整
    播完；False 表示被更新的請求取代或發生錯誤（error 非 None）。
    回呼都在 TTS 執行緒上執行，別做重活。
    """
    _worker.submit(text, device_name, rate, lang, on_start, on_done)
