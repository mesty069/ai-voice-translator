import os
import sys
from pathlib import Path


def register_cuda_dll_dirs():
    """把 nvidia-cublas / nvidia-cudnn 的 DLL 目錄加進 PATH。

    ctranslate2 用動態 LoadLibrary 找 cublas64_12.dll / cudnn64_9.dll，
    os.add_dll_directory 對它無效（實測），必須改 PATH。
    開發環境找 site-packages；PyInstaller 打包後找 _MEIPASS/nvidia。
    """
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
