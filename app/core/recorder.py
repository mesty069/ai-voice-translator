import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000

DEFAULT_DEVICE = "default"

# 只列 WASAPI host API 的裝置：每個實體裝置在 Windows 會被 MME / DirectSound /
# WASAPI / WDM-KS 各列一次，全部列出會重複。
# 必須用 WASAPI 而不是 MME：切換系統預設錄音裝置時 WinMM 會重排 MME 的
# 裝置編號，快取的索引會開到錯的裝置（麥克風隔離功能就是靠切換預設裝置）；
# WASAPI 的裝置識別穩定，不受影響。
_HOSTAPI = "Windows WASAPI"


def list_input_devices() -> list:
    """列出可選的錄音裝置名稱。"""
    names = []
    for dev in sd.query_devices():
        if (dev["max_input_channels"] > 0
                and sd.query_hostapis(dev["hostapi"])["name"] == _HOSTAPI):
            names.append(dev["name"])
    return names


def _default_wasapi_index():
    """把 PortAudio 的預設輸入裝置（MME）對應到同名的 WASAPI 裝置。

    MME 名稱是截斷到 31 字元的版本，用 prefix 比對。找不到時回傳 None
    （交給 PortAudio 用它的預設裝置）。"""
    try:
        default_idx = sd.default.device[0]
        if default_idx is None or default_idx < 0:
            return None
        default_name = sd.query_devices(default_idx)["name"]
    except Exception:
        return None
    for idx, dev in enumerate(sd.query_devices()):
        if (dev["max_input_channels"] > 0
                and sd.query_hostapis(dev["hostapi"])["name"] == _HOSTAPI
                and (dev["name"] == default_name
                     or dev["name"].startswith(default_name))):
            return idx
    return None


def resolve_input_device(name):
    """把設定裡的裝置名稱轉成 sounddevice 的 device index（WASAPI）。

    "default"（或空值）→ 對應系統預設輸入裝置的 WASAPI index，
    對不到時回傳 None（PortAudio 預設）。
    找不到指定名稱時丟 LookupError（裝置可能已拔除或改名）。
    """
    if not name or name == DEFAULT_DEVICE:
        return _default_wasapi_index()
    for idx, dev in enumerate(sd.query_devices()):
        if (dev["max_input_channels"] > 0
                and dev["name"] == name
                and sd.query_hostapis(dev["hostapi"])["name"] == _HOSTAPI):
            return idx
    raise LookupError(f"找不到錄音裝置「{name}」，請到設定頁重新選擇麥克風")


class Recorder:
    """按住期間從指定麥克風錄音，16kHz mono float32，可直接餵 faster-whisper。"""

    def __init__(self, device_name: str = DEFAULT_DEVICE, level_callback=None):
        self.device_name = device_name
        # 每收到一塊音訊就以該塊峰值（0.0~1.0）呼叫，在 PortAudio 執行緒執行，
        # 必須輕量且不可丟例外（丟了會中斷錄音串流）。
        self._level_callback = level_callback
        self._chunks = []
        self._stream = None
        self._lock = threading.Lock()

    def set_device(self, device_name: str):
        """更換錄音裝置，下一次 start() 生效。"""
        self.device_name = device_name

    def start(self):
        with self._lock:
            if self._stream is not None:
                return
            device = resolve_input_device(self.device_name)
            # WASAPI 共享模式只吃裝置原生取樣率，讓 Windows 轉成 16k；
            # 非 WASAPI 裝置（fallback）不能帶 WasapiSettings
            extra = None
            if device is not None and sd.query_hostapis(
                    sd.query_devices(device)["hostapi"])["name"] == _HOSTAPI:
                extra = sd.WasapiSettings(auto_convert=True)
            self._chunks = []
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device,
                extra_settings=extra,
                callback=self._callback,
            )
            self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        self._chunks.append(indata.copy())
        if self._level_callback is not None:
            self._level_callback(float(np.abs(indata).max()))

    def stop(self) -> np.ndarray:
        with self._lock:
            if self._stream is None:
                return np.zeros(0, dtype=np.float32)
            self._stream.stop()
            self._stream.close()
            self._stream = None
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(self._chunks, axis=0).flatten()
            self._chunks = []
            return audio

    @property
    def is_recording(self) -> bool:
        return self._stream is not None
