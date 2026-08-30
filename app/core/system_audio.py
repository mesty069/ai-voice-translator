import threading

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_DEVICE = "default"
BLOCK_FRAMES = 1600  # 0.1 秒


class RollingAudioBuffer:
    """連續音訊的滾動緩衝，用「絕對時間軸」定位。

    串流字幕每秒都要重新辨識「上一句句尾之後」的音訊，句尾位置來自
    辨識器回傳的時間戳（相對於送進去的那段音訊起點）。為了讓這個位置
    在多次 trim 之後仍然有意義，所有對外的秒數都是自緩衝建立起算的
    絕對秒：total_seconds 只增不減，start_seconds 是保留區的起點。
    thread-safe：擷取執行緒 append、引擎執行緒 since/trim。
    """

    def __init__(self, sample_rate=SAMPLE_RATE, max_seconds=60.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self._chunks = []
        self._kept = 0            # 保留區樣本數
        self._start = 0           # 保留區起點的絕對樣本數
        self._lock = threading.Lock()

    def append(self, frames):
        frames = np.asarray(frames, dtype=np.float32).reshape(-1)
        if len(frames) == 0:
            return
        with self._lock:
            self._chunks.append(frames)
            self._kept += len(frames)
            # 超過上限就丟最舊的，避免無限成長（引擎會定期 trim，這只是保險）
            overflow = self._kept - self.max_samples
            if overflow > 0:
                self._drop_locked(overflow)

    @property
    def total_seconds(self) -> float:
        with self._lock:
            return (self._start + self._kept) / self.sample_rate

    @property
    def start_seconds(self) -> float:
        with self._lock:
            return self._start / self.sample_rate

    def since(self, t_sec: float) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            data = np.concatenate(self._chunks)
            offset = int(round(t_sec * self.sample_rate)) - self._start
            offset = max(0, min(offset, len(data)))
            return data[offset:]

    def trim_before(self, t_sec: float):
        with self._lock:
            target = int(round(t_sec * self.sample_rate))
            drop = target - self._start
            if drop > 0:
                self._drop_locked(min(drop, self._kept))

    def tail_peak(self, seconds: float) -> float:
        with self._lock:
            if not self._chunks:
                return 0.0
            data = np.concatenate(self._chunks)
        n = int(seconds * self.sample_rate)
        tail = data[-n:] if n > 0 else data[:0]
        if len(tail) == 0:
            return 0.0
        return float(np.max(np.abs(tail)))

    def _drop_locked(self, n_samples: int):
        """丟掉最舊的 n 個樣本（呼叫端持鎖）。"""
        data = np.concatenate(self._chunks)[n_samples:]
        self._chunks = [data] if len(data) else []
        self._kept = len(data)
        self._start += n_samples


class SystemAudioCapture:
    """用 WASAPI loopback 擷取「電腦正在播放的聲音」（不含麥克風）。

    on_frames(np.ndarray)：每 0.1 秒一塊 float32 單聲道音框（在擷取執行緒上）。
    on_error(Exception)：擷取失敗時呼叫，之後擷取自行停止。
    """

    def __init__(self, device_name=DEFAULT_DEVICE, on_frames=None,
                 on_error=None):
        self.device_name = device_name
        self._on_frames = on_frames
        self._on_error = on_error
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
        # on_error 是在擷取執行緒上回呼的，呼叫端會反手 stop() 擷取，
        # 這時 join 自己會丟 RuntimeError；此時只要把旗標放掉就好。
        if thread is not None and thread is not threading.current_thread():
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
                    if self._on_frames is not None:
                        self._on_frames(np.asarray(frames, dtype=np.float32))
        except Exception as e:
            self._running = False
            if self._on_error is not None:
                self._on_error(e)
