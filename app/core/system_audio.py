import threading

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_DEVICE = "default"
BLOCK_FRAMES = 1600  # 0.1 秒


class SegmentAccumulator:
    """把連續音框切成一句一句。純邏輯、無音訊裝置相依，方便測試。

    規則：偵測到語音才開始收；語音後靜音達 silence_ms 就送出；
    連續語音超過 max_seconds 強制切段（切完仍視為語音進行中）；
    長度不足 min_seconds 的段落丟棄（雜訊、短促聲響）。
    """

    def __init__(self, sample_rate=SAMPLE_RATE, silence_ms=600,
                 max_seconds=8.0, min_seconds=0.8, threshold=0.008):
        self.sample_rate = sample_rate
        self.silence_samples = int(sample_rate * silence_ms / 1000)
        self.max_samples = int(sample_rate * max_seconds)
        self.min_samples = int(sample_rate * min_seconds)
        self.threshold = threshold
        self._buf = []
        self._buf_len = 0
        self._silence_run = 0
        self._has_speech = False

    def push(self, frames) -> list:
        frames = np.asarray(frames, dtype=np.float32).reshape(-1)
        if len(frames) == 0:
            return []
        peak = float(np.abs(frames).max())
        is_speech = peak >= self.threshold
        if is_speech:
            self._has_speech = True
            self._silence_run = 0
        elif self._has_speech:
            self._silence_run += len(frames)
        if not self._has_speech:
            return []
        if is_speech:
            self._buf.append(frames)
            self._buf_len += len(frames)
        if self._silence_run >= self.silence_samples:
            seg = self._flush(keep_speech=False)
            return [seg] if seg is not None else []
        if self._buf_len >= self.max_samples:
            seg = self._flush(keep_speech=True)
            return [seg] if seg is not None else []
        return []

    def drain(self):
        """停止擷取時把還沒送出的尾巴取出。"""
        return self._flush(keep_speech=False)

    def _flush(self, keep_speech):
        buf, length = self._buf, self._buf_len
        self._buf, self._buf_len = [], 0
        self._silence_run = 0
        self._has_speech = keep_speech
        if not buf or length < self.min_samples:
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
            thread.join(timeout=2.0)

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
        except Exception as e:
            self._running = False
            if self._on_error is not None:
                self._on_error(e)
