import threading

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_DEVICE = "default"
BLOCK_FRAMES = 1600  # 0.1 秒


class SegmentAccumulator:
    """把連續音框切成一句一句。純邏輯、無音訊裝置相依，方便測試。

    刻意不用「絕對音量門檻」：桌面環境常有持續背景音（其他程式的聲音、
    麥克風監聽回送），實測背景 RMS 可達 0.03 以上，絕對門檻會讓程式
    永遠認為「一直有人在講話」而切不出任何一段。

    改用兩個判準：
    - 停頓＝音量低於「本段目前最大音量」的 quiet_ratio 倍（相對判斷，
      不受背景音絕對大小影響，影片播到一半才開始擷取也成立）
    - 純數位靜音＝整段最大音量都低於 min_peak，直接丟棄
    另外累計「非停頓」的長度，太短的段落（雜訊、短促聲響）丟棄。
    """

    def __init__(self, sample_rate=SAMPLE_RATE, silence_ms=600,
                 max_seconds=8.0, min_seconds=0.8,
                 quiet_ratio=0.2, min_peak=0.01):
        self.sample_rate = sample_rate
        self.silence_samples = int(sample_rate * silence_ms / 1000)
        self.max_samples = int(sample_rate * max_seconds)
        self.min_samples = int(sample_rate * min_seconds)
        self.quiet_ratio = quiet_ratio
        self.min_peak = min_peak
        self._buf = []
        self._buf_len = 0
        self._quiet_run = 0
        self._speech_len = 0
        self._peak = 0.0

    @staticmethod
    def _level(frames) -> float:
        """用 RMS 而非尖峰值：對雜訊比較不敏感。"""
        return float(np.sqrt(np.mean(frames.astype(np.float64) ** 2)))

    def push(self, frames) -> list:
        frames = np.asarray(frames, dtype=np.float32).reshape(-1)
        if len(frames) == 0:
            return []
        level = self._level(frames)
        self._peak = max(self._peak, level)
        # 一律收進緩衝：段落內的短停頓要保留，語音才連續、辨識才準
        self._buf.append(frames)
        self._buf_len += len(frames)
        if level < self._peak * self.quiet_ratio:
            self._quiet_run += len(frames)
        else:
            self._quiet_run = 0
            self._speech_len += len(frames)
        if self._quiet_run >= self.silence_samples:
            seg = self._flush()
            return [seg] if seg is not None else []
        if self._buf_len >= self.max_samples:
            seg = self._flush()
            return [seg] if seg is not None else []
        return []

    def drain(self):
        """停止擷取時把還沒送出的尾巴取出。"""
        return self._flush()

    def _flush(self):
        buf, speech_len, peak = self._buf, self._speech_len, self._peak
        self._buf, self._buf_len = [], 0
        self._quiet_run = 0
        self._speech_len = 0
        self._peak = 0.0
        if not buf:
            return None
        if peak < self.min_peak:          # 純數位靜音
            return None
        if speech_len < self.min_samples:  # 有聲部分太短
            return None
        return np.concatenate(buf)


class SystemAudioCapture:
    """用 WASAPI loopback 擷取「電腦正在播放的聲音」（不含麥克風）。

    on_segment(np.ndarray)：切出一段語音時呼叫（在擷取執行緒上）。
    on_error(Exception)：擷取失敗時呼叫，之後擷取自行停止。
    """

    def __init__(self, device_name=DEFAULT_DEVICE, on_segment=None,
                 on_error=None, silence_ms=600, max_seconds=8.0):
        self.device_name = device_name
        self._on_segment = on_segment
        self._on_error = on_error
        self._acc = SegmentAccumulator(
            silence_ms=silence_ms, max_seconds=max_seconds)
        self._thread = None
        self._running = False

    @staticmethod
    def list_output_devices() -> list:
        import soundcard as sc
        return [str(s.name) for s in sc.all_speakers()]

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="system-audio")
        self._thread.start()

    def stop(self):
        self._running = False
        thread, self._thread = self._thread, None
        if thread is not None:
            # 讀取區塊只有 0.1 秒，稍等即可收屍；等太久會卡住呼叫端的 GUI 執行緒
            thread.join(timeout=0.2)

    def _resolve_speaker(self):
        import soundcard as sc
        if not self.device_name or self.device_name == DEFAULT_DEVICE:
            return sc.default_speaker()
        for speaker in sc.all_speakers():
            if str(speaker.name) == self.device_name:
                return speaker
        raise LookupError(f"找不到輸出裝置「{self.device_name}」")

    def _run(self):
        import soundcard as sc
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
        try:
            speaker = self._resolve_speaker()
            loopback = sc.get_microphone(
                str(speaker.name), include_loopback=True)
            with loopback.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
                while self._running:
                    data = rec.record(numframes=BLOCK_FRAMES)
                    frames = data[:, 0] if getattr(data, "ndim", 1) > 1 else data
                    for segment in self._acc.push(frames):
                        if self._on_segment is not None:
                            self._on_segment(segment)
            # 正常停止：把最後一句還沒遇到停頓的尾段補送出去
            tail = self._acc.drain()
            if tail is not None and self._on_segment is not None:
                self._on_segment(tail)
        except Exception as e:
            self._running = False
            if self._on_error is not None:
                self._on_error(e)
