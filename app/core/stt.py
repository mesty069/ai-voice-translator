import threading

import numpy as np

from .cuda_dlls import register_cuda_dll_dirs


class SpeechToText:
    """faster-whisper 中文語音辨識。模型載入很慢，需在背景執行緒呼叫 load()。"""

    def __init__(self, model_size: str = "small", device: str = "cpu",
                 compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()

    @staticmethod
    def _lower_priority():
        """載入 3GB 模型會重度讀碟＋吃 CPU，降優先權避免前景程式卡頓。
        回傳恢復函式。"""
        try:
            import psutil
            proc = psutil.Process()
            original = proc.nice()
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return lambda: proc.nice(original)
        except Exception:
            return lambda: None

    def load(self):
        from faster_whisper import WhisperModel
        register_cuda_dll_dirs()
        restore_priority = self._lower_priority()
        try:
            with self._lock:
                try:
                    model = WhisperModel(
                        self.model_size, device=self.device,
                        compute_type=self.compute_type)
                    self._warmup(model)
                except Exception:
                    if self.device == "cpu":
                        raise
                    # GPU 不可用（缺 CUDA 函式庫、顯示卡被占用等）→ 退回 CPU
                    self.device = "cpu"
                    self.compute_type = "int8"
                    model = WhisperModel(
                        self.model_size, device="cpu", compute_type="int8")
                    self._warmup(model)
                self._model = model
        finally:
            try:
                restore_priority()
            except Exception:
                pass

    @staticmethod
    def _warmup(model):
        """用短暫靜音跑一次推論：確認 CUDA 函式庫真的能用（載入成功不代表
        推論成功），同時完成首次推論的初始化，讓第一句真正的語音不用等。"""
        silence = np.zeros(8000, dtype=np.float32)
        segments, _info = model.transcribe(silence, language="zh", beam_size=1)
        for _ in segments:
            pass

    def reload(self, model_size: str):
        with self._lock:
            self._model = None
        self.model_size = model_size
        self.load()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray, language: str = "zh",
               beam_size: int = 5) -> str:
        # beam_size=5 給麥克風翻譯（一句話等結果，要準）；
        # 即時字幕用 1（貪婪解碼），快 2–3 倍，字幕場景看不出差別
        # 麥克風與系統字幕兩條管線共用同一個模型，必須序列化，
        # 否則會同時擠 GPU 造成拖慢甚至崩潰。
        # 檢查與取用都要在鎖內、且取本地變數：在鎖外檢查的話，
        # reload() 會在檢查後把 _model 清成 None，進了鎖就 AttributeError。
        with self._lock:
            model = self._model
            if model is None:
                raise RuntimeError("語音模型尚未載入完成")
            segments, _info = model.transcribe(
                audio,
                language=language,
                beam_size=beam_size,
                vad_filter=True,
                initial_prompt="以下是繁體中文的句子。" if language == "zh" else None,
            )
            return "".join(seg.text for seg in segments).strip()
