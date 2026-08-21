import os
import tempfile
import threading

import sounddevice as sd
import soundfile as sf


_lock = threading.Lock()


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


def speak(text: str, device_name: str = "default"):
    """用 Windows 內建語音把英文唸出來，可指定輸出裝置。阻塞直到播完。"""
    import pyttsx3
    with _lock:
        wav_path = os.path.join(
            tempfile.gettempdir(), "ai_translator_tts.wav")
        engine = pyttsx3.init()
        for voice in engine.getProperty("voices"):
            vid = voice.id.upper()
            if "EN-US" in vid or "EN_US" in vid or "ENGLISH" in voice.name.upper():
                engine.setProperty("voice", voice.id)
                break
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        data, samplerate = sf.read(wav_path)
        sd.play(data, samplerate, device=_resolve_device(device_name))
        sd.wait()
