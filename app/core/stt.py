import os
import threading
from pathlib import Path

import numpy as np


def _register_cuda_dll_dirs():
    """把 nvidia-cublas / nvidia-cudnn DLL 目錄加進 PATH，
    讓 ctranslate2 的 CUDA 模式找得到 cublas64_12.dll / cudnn64_9.dll。
    （os.add_dll_directory 對 ctranslate2 的動態 LoadLibrary 無效，實測
    必須用 PATH。）開發環境找 site-packages；打包後找 _internal/nvidia。"""
    import sys
    roots = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass) / "nvidia")
    else:
        import site
        roots.extend(Path(sp) / "nvidia" for sp in site.getsitepackages())
    bin_dirs = []
    for nvidia_dir in roots:
        if nvidia_dir.is_dir():
            bin_dirs.extend(str(p) for p in nvidia_dir.glob("*/bin"))
    if bin_dirs:
        os.environ["PATH"] = (
            os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", ""))


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
        _register_cuda_dll_dirs()
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

    def transcribe(self, audio: np.ndarray, language: str = "zh") -> str:
        if self._model is None:
            raise RuntimeError("語音模型尚未載入完成")
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
            initial_prompt="以下是繁體中文的句子。" if language == "zh" else None,
        )
        return "".join(seg.text for seg in segments).strip()
