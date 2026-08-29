# 系統聲音即時雙語字幕 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 擷取電腦播放的聲音（不含麥克風），即時辨識並用本機模型翻成母語，以可拖動的雙語懸浮字幕顯示，且與麥克風字幕明顯區分、重疊時互相推開。

**Architecture:** `soundcard` 做 WASAPI loopback 擷取 → 能量門檻切句 → 與麥克風共用的 faster-whisper 辨識（加鎖序列化）→ CTranslate2 本機翻譯 → Qt 訊號 → 專屬字幕浮層。翻譯模型（NLLB-600M / NLLB-1.3B / OPUS-MT）於設定頁切換，首次使用才下載。

**Tech Stack:** Python 3.12、PySide6 + qfluentwidgets、soundcard（新增）、faster-whisper、CTranslate2、transformers tokenizer（新增）、huggingface_hub

## Global Constraints

- 平台限定 Windows 10/11；音訊擷取用 WASAPI loopback。
- 翻譯**不得呼叫 DeepSeek**，一律本機模型（DeepSeek 仍供麥克風翻譯使用）。
- 新增相依**不得引入 `torch`**（會讓打包從 2.4GB 膨脹到 5GB 以上）。
- 所有使用者可見文字用繁體中文。
- 語音辨識模型（`SpeechToText`）與麥克風功能**共用同一個實例**，不得重複載入。
- 麥克風錄音/處理期間優先，系統字幕的辨識必須讓路。
- 測試指令一律 `.venv\Scripts\python.exe -m pytest tests -q`；每個任務結束前整包測試必須全綠。
- 開始本計畫時既有測試為 20 個，全部必須持續通過。
- 已驗證可用的模型 repo（勿更換）：
  - `entai2965/nllb-200-distilled-600M-ctranslate2`
  - `entai2965/nllb-200-distilled-1.3B-ctranslate2`
  - `gaudi/opus-mt-{src}-{tgt}-ctranslate2`
  三者都內含 tokenizer 檔案，不需另外抓原始模型。

## File Structure

| 檔案 | 責任 |
|---|---|
| `app/core/system_audio.py`（新） | loopback 擷取 + 能量切句。`SegmentAccumulator` 為純邏輯可單測；`SystemAudioCapture` 負責執行緒與裝置。 |
| `app/core/local_translate.py`（新） | CTranslate2 本機翻譯、模型登錄表、語言代碼映射、下載與 GPU/CPU 退回。 |
| `app/core/system_captions.py`（新） | 編排：語音段 → 辨識 → 翻譯 → 發訊號。唯一知道「麥克風優先」規則的地方。 |
| `app/ui/overlay_base.py`（新） | 浮層共用行為：拖動、邊框縮放、位置/大小記憶、透明度。 |
| `app/ui/system_subtitle.py`（新） | 系統聲音雙語字幕浮層（青藍配色、🔊 標籤、歷史）。 |
| `app/ui/subtitle.py`（改） | 改為繼承 `DraggableResizableOverlay`，刪除重複的拖動/縮放程式碼。 |
| `app/core/stt.py`（改） | `transcribe()` 加鎖，支援兩條管線呼叫。 |
| `app/config.py`（改） | 新增 `system_captions` 預設區段。 |
| `app/controller.py`（改） | 系統字幕熱鍵監聽 + `mic_busy()` 查詢。 |
| `app/ui/main_window.py`（改） | 建立編排器與浮層、三處開關同步、重疊推開、結束時收拾。 |
| `app/ui/bubble.py`（改） | 長按選單新增「🔊 系統字幕 開/關」。 |
| `app/ui/settings_page.py`（改） | 新增「系統聲音字幕」設定區塊。 |
| `tests/test_system_audio.py`（新） | 切句邏輯單元測試。 |
| `tests/test_local_translate.py`（新） | 模型登錄表與語言代碼映射單元測試。 |
| `tests/test_overlay_geometry.py`（新） | 重疊推開幾何單元測試。 |

---

### Task 1: 設定區段與相依套件

**Files:**
- Modify: `app/config.py`（`DEFAULT_CONFIG`）
- Modify: `requirements.txt`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 無
- Produces: `config.get("system_captions", <key>)` 可讀到 `enabled=False`、`hotkey_type="keyboard"`、`hotkey_key="f11"`、`device="default"`、`language=""`、`engine="nllb-600m"`、`compute_device="auto"`、`font_size=20`、`opacity=100`、`bg_color="#0d2b33"`、`segment_silence_ms=600`、`max_segment_sec=8`

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_config.py` 末尾加入：

```python
def test_system_captions_defaults(tmp_path):
    from app.config import Config
    cfg = Config(tmp_path / "config.json")
    assert cfg.get("system_captions", "enabled") is False
    assert cfg.get("system_captions", "hotkey_key") == "f11"
    assert cfg.get("system_captions", "engine") == "nllb-600m"
    assert cfg.get("system_captions", "language") == ""
    assert cfg.get("system_captions", "compute_device") == "auto"
    assert cfg.get("system_captions", "segment_silence_ms") == 600


def test_reset_restores_system_captions(tmp_path):
    from app.config import Config
    cfg = Config(tmp_path / "config.json")
    cfg.set("system_captions", "enabled", True)
    cfg.set("system_captions", "engine", "opus-mt")
    cfg.reset_to_defaults()
    assert cfg.get("system_captions", "enabled") is False
    assert cfg.get("system_captions", "engine") == "nllb-600m"
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -q`
Expected: FAIL，`assert None is False`（`system_captions` 區段不存在）

- [ ] **Step 3: 加入預設設定**

在 `app/config.py` 的 `DEFAULT_CONFIG` 中、`"ui"` 區段之前插入：

```python
    "system_captions": {
        "enabled": False,
        "hotkey_type": "keyboard",
        "hotkey_key": "f11",
        "device": "default",
        "language": "",            # 空字串＝跟隨目標語言
        "engine": "nllb-600m",     # nllb-600m | nllb-1.3b | opus-mt
        "compute_device": "auto",  # auto | cpu
        "font_size": 20,
        "opacity": 100,
        "bg_color": "#0d2b33",
        "segment_silence_ms": 600,
        "max_segment_sec": 8,
    },
```

- [ ] **Step 4: 執行測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS，22 passed

- [ ] **Step 5: 安裝並記錄新相依**

Run: `.venv/Scripts/python.exe -m pip install soundcard transformers sentencepiece`

在 `requirements.txt` 的 `pyttsx3` 之後加入：

```
# 系統聲音擷取（WASAPI loopback）
soundcard
# 本機翻譯用的 tokenizer（不會安裝 torch）
transformers
sentencepiece
# 簡體→台灣繁體（NLLB 以簡體解碼品質才完整）
opencc-python-reimplemented
```

驗證沒有意外安裝 torch：

Run: `.venv/Scripts/python.exe -c "import importlib.util; print('torch:', importlib.util.find_spec('torch') is not None)"`
Expected: `torch: False`

- [ ] **Step 6: Commit**

```bash
git add app/config.py requirements.txt tests/test_config.py
git commit -m "feat: 新增系統聲音字幕的設定區段與相依套件"
```

---

### Task 2: 系統聲音擷取與切句

**Files:**
- Create: `app/core/system_audio.py`
- Test: `tests/test_system_audio.py`

**Interfaces:**
- Consumes: 無
- Produces:
  - `SAMPLE_RATE = 16000`、`DEFAULT_DEVICE = "default"`、`BLOCK_FRAMES = 1600`
  - `SegmentAccumulator(sample_rate=16000, silence_ms=600, max_seconds=8.0, min_seconds=0.8, quiet_ratio=0.2, min_peak=0.01)`；`push(frames) -> list[np.ndarray]`、`drain() -> np.ndarray | None`
  - `SystemAudioCapture(device_name="default", on_segment=None, on_error=None, silence_ms=600, max_seconds=8.0)`；`start()`、`stop()`、`is_running`、靜態 `list_output_devices() -> list[str]`

- [ ] **Step 1: 寫失敗測試**

建立 `tests/test_system_audio.py`：

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.system_audio import SAMPLE_RATE, SegmentAccumulator


def _speech(seconds):
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def _feed(acc, audio, block=1600):
    out = []
    for i in range(0, len(audio), block):
        out.extend(acc.push(audio[i:i + block]))
    return out


def test_silence_produces_nothing():
    acc = SegmentAccumulator()
    assert _feed(acc, _silence(3.0)) == []


def test_speech_then_silence_emits_one_segment():
    acc = SegmentAccumulator(silence_ms=600, min_seconds=0.8)
    segments = _feed(acc, np.concatenate([_speech(2.0), _silence(1.0)]))
    assert len(segments) == 1
    assert len(segments[0]) >= int(SAMPLE_RATE * 2.0)


def test_short_blip_is_discarded():
    acc = SegmentAccumulator(silence_ms=600, min_seconds=0.8)
    assert _feed(acc, np.concatenate([_speech(0.3), _silence(1.0)])) == []


def test_long_speech_is_force_cut():
    acc = SegmentAccumulator(silence_ms=600, max_seconds=4.0)
    segments = _feed(acc, _speech(9.0))
    assert len(segments) >= 2
    for seg in segments:
        assert len(seg) <= int(SAMPLE_RATE * 4.0) + 1600


def test_speech_after_force_cut_still_accumulates():
    acc = SegmentAccumulator(silence_ms=600, max_seconds=2.0, min_seconds=0.5)
    first = _feed(acc, _speech(5.0))
    rest = _feed(acc, _silence(1.0))
    assert len(first) >= 2
    assert len(rest) == 1


def test_drain_returns_pending_audio():
    acc = SegmentAccumulator(min_seconds=0.5)
    acc.push(_speech(1.0))
    tail = acc.drain()
    assert tail is not None
    assert len(tail) >= int(SAMPLE_RATE * 1.0)


def _background(seconds, amplitude=0.03):
    """模擬桌面持續背景音（實測本機為 0.03 以上）。"""
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 60 * t)).astype(np.float32)


def test_constant_background_does_not_block_segmentation():
    """回歸測試：持續背景音高於任何固定門檻時，仍要切得出段落。"""
    acc = SegmentAccumulator(silence_ms=600, min_seconds=0.8)
    audio = np.concatenate([_background(1.0), _speech(2.0), _background(1.5)])
    segments = _feed(acc, audio)
    assert len(segments) == 1


def test_pure_silence_segment_is_discarded_on_force_cut():
    """整段都是數位靜音時，即使觸發強制切段也不該送出。"""
    acc = SegmentAccumulator(silence_ms=600, max_seconds=2.0, min_seconds=0.5)
    assert _feed(acc, _silence(6.0)) == []
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_system_audio.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.core.system_audio'`

- [ ] **Step 3: 實作模組**

建立 `app/core/system_audio.py`：

```python
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
```

- [ ] **Step 4: 執行測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS，30 passed

- [ ] **Step 5: 實機驗證真的收得到系統聲音**

Run:
```bash
.venv/Scripts/python.exe -X utf8 -c "import time; import numpy as np; import sounddevice as sd; from app.core.system_audio import SystemAudioCapture; got=[]; cap=SystemAudioCapture(on_segment=got.append, on_error=lambda e: print('ERR', e)); cap.start(); sr=48000; tone=(0.3*np.sin(2*np.pi*440*np.arange(sr*3)/sr)).astype('float32'); sd.play(tone, sr); sd.wait(); sd.stop(); time.sleep(1.5); cap.stop(); print('收到段數:', len(got), '首段秒數:', round(len(got[0])/16000,2) if got else 0)"
```
Expected: `收到段數: 1`（或以上），首段秒數約 3

- [ ] **Step 6: Commit**

```bash
git add app/core/system_audio.py tests/test_system_audio.py
git commit -m "feat: 新增系統聲音 loopback 擷取與切句"
```

---

### Task 3: 本機翻譯引擎

**Files:**
- Create: `app/core/local_translate.py`
- Test: `tests/test_local_translate.py`

**Interfaces:**
- Consumes: `app.config.LANGUAGES`（既有，`[(code, 顯示名)]`）
- Produces:
  - `ENGINES: dict[str, dict]`、`ENGINE_LABELS: list[tuple[str, str]]`（給設定頁下拉用，格式 `[(engine_id, 顯示名)]`）
  - `NLLB_CODES: dict[str, str]`
  - `repo_for(engine, src, tgt) -> tuple[str, str]`（回傳 `(repo_id, kind)`，kind 為 `"nllb"` 或 `"opus"`）
  - `LocalTranslator(engine="nllb-600m", compute_device="auto")`；`ensure_loaded(src, tgt, progress_cb=None)`、`translate(text, src, tgt) -> str`、`is_ready(src, tgt) -> bool`、`set_engine(engine)`

- [ ] **Step 1: 寫失敗測試**

建立 `tests/test_local_translate.py`：

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import LANGUAGES
from app.core.local_translate import (
    ENGINES,
    ENGINE_LABELS,
    NLLB_CODES,
    LocalTranslator,
    repo_for,
)


def test_nllb_codes_cover_all_app_languages():
    for code, _name in LANGUAGES:
        assert code in NLLB_CODES, f"缺少 {code} 的 NLLB 代碼"


def test_engine_labels_match_engines():
    assert {e for e, _ in ENGINE_LABELS} == set(ENGINES)


def test_repo_for_nllb_is_fixed():
    repo, kind = repo_for("nllb-600m", "en", "zh")
    assert repo == "entai2965/nllb-200-distilled-600M-ctranslate2"
    assert kind == "nllb"


def test_repo_for_opus_is_language_pair_specific():
    repo, kind = repo_for("opus-mt", "en", "zh")
    assert repo == "gaudi/opus-mt-en-zh-ctranslate2"
    assert kind == "opus"


def test_repo_for_unknown_engine_raises():
    with pytest.raises(KeyError):
        repo_for("nope", "en", "zh")


def test_translate_empty_text_returns_empty_without_loading():
    translator = LocalTranslator()
    assert translator.translate("   ", "en", "zh") == ""
    assert not translator.is_ready("en", "zh")


def test_set_engine_invalidates_loaded_model():
    translator = LocalTranslator()
    translator._key = ("nllb-600m", "en", "zh")  # 假裝已載入
    assert translator.is_ready("en", "zh")
    translator.set_engine("opus-mt")
    assert not translator.is_ready("en", "zh")
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_local_translate.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.core.local_translate'`

- [ ] **Step 3: 實作模組**

建立 `app/core/local_translate.py`：

```python
import threading

# 程式的語言代碼 → NLLB-200 的語言代碼
# 中文刻意用簡體 zho_Hans 解碼再以 OpenCC 轉台灣繁體：NLLB 的繁體訓練資料
# 遠少於簡體，直接用 zho_Hant 會系統性地在逗號後截斷（實測 6 句中 3 句），
# 換 zho_Hans 全部完整。
NLLB_CODES = {
    "zh": "zho_Hans",
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "ru": "rus_Cyrl",
}

# 已驗證存在且內含 tokenizer 的 CTranslate2 模型
ENGINES = {
    "nllb-600m": {
        "kind": "nllb",
        "repo": "entai2965/nllb-200-distilled-600M-ctranslate2",
    },
    "nllb-1.3b": {
        "kind": "nllb",
        "repo": "entai2965/nllb-200-distilled-1.3B-ctranslate2",
    },
    "opus-mt": {
        "kind": "opus",
        "repo": "gaudi/opus-mt-{src}-{tgt}-ctranslate2",
    },
}

ENGINE_LABELS = [
    ("nllb-600m", "NLLB 600M（約 600MB，通用、快）"),
    ("nllb-1.3b", "NLLB 1.3B（約 1.3GB，品質最好、較慢）"),
    ("opus-mt", "OPUS-MT（約 80MB，單一語言對、最快）"),
]


def repo_for(engine: str, src: str, tgt: str):
    """回傳 (repo_id, kind)。engine 不存在時丟 KeyError。"""
    spec = ENGINES[engine]
    return spec["repo"].format(src=src, tgt=tgt), spec["kind"]


class LocalTranslator:
    """CTranslate2 本機翻譯。模型首次使用才下載，之後快取在本機。

    同一個實例可換引擎/語言對，換了會重新載入。所有公開方法皆執行緒安全。
    """

    def __init__(self, engine: str = "nllb-600m",
                 compute_device: str = "auto"):
        self.engine = engine
        self.compute_device = compute_device
        self._lock = threading.RLock()
        self._key = None          # (engine, src, tgt)
        self._translator = None
        self._tokenizer = None
        self._kind = None

    def set_engine(self, engine: str):
        with self._lock:
            if engine != self.engine:
                self.engine = engine
                self._unload()

    def set_compute_device(self, device: str):
        with self._lock:
            if device != self.compute_device:
                self.compute_device = device
                self._unload()

    def is_ready(self, src: str, tgt: str) -> bool:
        return self._key == (self.engine, src, tgt)

    def _unload(self):
        self._key = None
        self._translator = None
        self._tokenizer = None
        self._kind = None

    def _device_candidates(self):
        """回傳 [(device, compute_type)]，依序嘗試。"""
        if self.compute_device == "cpu":
            return [("cpu", "int8")]
        return [("cuda", "int8_float16"), ("cpu", "int8")]

    def ensure_loaded(self, src: str, tgt: str, progress_cb=None):
        with self._lock:
            if self.is_ready(src, tgt):
                return
            repo, kind = repo_for(self.engine, src, tgt)
            if progress_cb is not None:
                progress_cb(f"正在準備翻譯模型（{repo}）…")
            from huggingface_hub import snapshot_download
            path = snapshot_download(repo)

            import ctranslate2
            from transformers import AutoTokenizer

            last_error = None
            translator = None
            for device, compute_type in self._device_candidates():
                try:
                    translator = ctranslate2.Translator(
                        path, device=device, compute_type=compute_type)
                    break
                except Exception as e:  # GPU 不可用或記憶體不足 → 退回 CPU
                    last_error = e
            if translator is None:
                raise RuntimeError(f"翻譯模型載入失敗：{last_error}")

            tokenizer_kwargs = {}
            if kind == "nllb":
                tokenizer_kwargs["src_lang"] = NLLB_CODES[src]
            self._tokenizer = AutoTokenizer.from_pretrained(
                path, **tokenizer_kwargs)
            self._translator = translator
            self._kind = kind
            self._key = (self.engine, src, tgt)

    def translate(self, text: str, src: str, tgt: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        self.ensure_loaded(src, tgt)
        with self._lock:
            tokenizer = self._tokenizer
            source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
            if self._kind == "nllb":
                results = self._translator.translate_batch(
                    [source], target_prefix=[[NLLB_CODES[tgt]]], beam_size=2)
                hypothesis = results[0].hypotheses[0][1:]  # 去掉語言標記
            else:
                results = self._translator.translate_batch(
                    [source], beam_size=2)
                hypothesis = results[0].hypotheses[0]
            ids = tokenizer.convert_tokens_to_ids(hypothesis)
            return tokenizer.decode(ids, skip_special_tokens=True).strip()
```

- [ ] **Step 4: 執行測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS，37 passed

- [ ] **Step 5: 實機驗證翻譯品質（會下載約 600MB，只做一次）**

Run:
```bash
.venv/Scripts/python.exe -X utf8 -c "import truststore; truststore.inject_into_ssl(); import time; from app.core.local_translate import LocalTranslator; t=LocalTranslator(); s=time.time(); t.ensure_loaded('en','zh', print); print('載入秒數', round(time.time()-s,1)); s=time.time(); out=t.translate('The meeting will start in five minutes, please join now.', 'en', 'zh'); print('翻譯:', out); print('耗時秒', round(time.time()-s,2))"
```
Expected: 印出中文翻譯（例如「會議將在五分鐘後開始，請立即加入。」），單句耗時 < 1 秒

- [ ] **Step 6: Commit**

```bash
git add app/core/local_translate.py tests/test_local_translate.py
git commit -m "feat: 新增 CTranslate2 本機翻譯引擎"
```

---

### Task 4: 語音模型併發鎖

**Files:**
- Modify: `app/core/stt.py`（`SpeechToText.transcribe`）
- Test: `tests/test_stt_lock.py`（新）

**Interfaces:**
- Consumes: 既有 `SpeechToText`
- Produces: `SpeechToText.transcribe(audio, language="zh")` 為執行緒安全，同時間只有一個呼叫在跑模型

- [ ] **Step 1: 寫失敗測試**

建立 `tests/test_stt_lock.py`：

```python
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.stt import SpeechToText


class _SlowFakeModel:
    """記錄是否有兩個 transcribe 同時進行。"""

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def transcribe(self, audio, **kwargs):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.15)
        with self._guard:
            self.active -= 1
        return ([], None)


def test_transcribe_is_serialized():
    stt = SpeechToText()
    fake = _SlowFakeModel()
    stt._model = fake
    audio = np.zeros(1600, dtype=np.float32)

    threads = [threading.Thread(target=stt.transcribe, args=(audio,))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert fake.max_active == 1, "transcribe 沒有序列化，兩條管線會同時擠 GPU"
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stt_lock.py -q`
Expected: FAIL，`assert 4 == 1`（或 2/3，總之大於 1）

- [ ] **Step 3: 加鎖**

修改 `app/core/stt.py` 的 `transcribe`，把模型呼叫包進既有的 `self._lock`：

```python
    def transcribe(self, audio: np.ndarray, language: str = "zh") -> str:
        if self._model is None:
            raise RuntimeError("語音模型尚未載入完成")
        # 麥克風與系統字幕兩條管線共用同一個模型，必須序列化，
        # 否則會同時擠 GPU 造成拖慢甚至崩潰
        with self._lock:
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,
                initial_prompt="以下是繁體中文的句子。" if language == "zh" else None,
            )
            return "".join(seg.text for seg in segments).strip()
```

- [ ] **Step 4: 執行測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS，38 passed

- [ ] **Step 5: 確認麥克風流程沒被拖慢**

Run:
```bash
.venv/Scripts/python.exe -X utf8 -c "import time; import numpy as np; from app.core.stt import SpeechToText; s=SpeechToText('large-v3','cuda','int8_float16'); s.load(); t=time.time(); s.transcribe(np.zeros(32000, dtype=np.float32)); print('辨識 2 秒音訊耗時', round(time.time()-t,2), '秒')"
```
Expected: 約 0.3 秒（與加鎖前相同）

- [ ] **Step 6: Commit**

```bash
git add app/core/stt.py tests/test_stt_lock.py
git commit -m "fix: 語音辨識加鎖，支援麥克風與系統字幕共用模型"
```

---

### Task 5: 系統字幕編排器

**Files:**
- Create: `app/core/system_captions.py`
- Test: `tests/test_system_captions.py`（新）

**Interfaces:**
- Consumes: `SystemAudioCapture`（Task 2）、`LocalTranslator`（Task 3）、`SpeechToText.transcribe`（Task 4）、`Config`
- Produces: `SystemCaptionsController(config, stt, mic_busy, parent=None)`（QObject）
  - 訊號 `caption_ready(str, str)`、`state_changed(str, str)`、`error_occurred(str)`
  - 方法 `start()`、`stop()`、`is_running`（property）
  - `mic_busy` 是回傳 bool 的無參數函式；為 True 時延後辨識（最多等 10 秒）

- [ ] **Step 1: 寫失敗測試**

建立 `tests/test_system_captions.py`：

```python
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config
from app.core.system_captions import SystemCaptionsController


class _FakeStt:
    def __init__(self):
        self.languages = []

    def transcribe(self, audio, language="zh"):
        self.languages.append(language)
        return "hello world"


class _FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, text, src, tgt):
        self.calls.append((text, src, tgt))
        return "你好世界"

    def set_engine(self, engine):
        pass

    def set_compute_device(self, device):
        pass


def _controller(tmp_path, mic_busy=lambda: False):
    cfg = Config(tmp_path / "config.json")
    cfg.set("language", "source", "zh")
    cfg.set("language", "target", "en")
    ctrl = SystemCaptionsController(cfg, _FakeStt(), mic_busy)
    ctrl._translator = _FakeTranslator()
    return cfg, ctrl


def test_segment_produces_bilingual_caption(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    captions = []
    ctrl.caption_ready.connect(lambda a, b: captions.append((a, b)))
    ctrl._process(np.zeros(16000, dtype=np.float32))
    assert captions == [("hello world", "你好世界")]


def test_uses_target_language_for_recognition_and_native_for_translation(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    ctrl._process(np.zeros(16000, dtype=np.float32))
    assert ctrl.stt.languages == ["en"]          # 系統聲音＝目標語言
    assert ctrl._translator.calls[0][1:] == ("en", "zh")  # 翻成母語


def test_explicit_language_overrides_target(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    cfg.set("system_captions", "language", "ja")
    ctrl._process(np.zeros(16000, dtype=np.float32))
    assert ctrl.stt.languages == ["ja"]
    assert ctrl._translator.calls[0][1:] == ("ja", "zh")


def test_empty_transcription_emits_nothing(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    ctrl.stt.transcribe = lambda audio, language="zh": "   "
    captions = []
    ctrl.caption_ready.connect(lambda a, b: captions.append((a, b)))
    ctrl._process(np.zeros(16000, dtype=np.float32))
    assert captions == []


def test_waits_while_microphone_is_busy(tmp_path):
    busy = {"value": True}
    cfg, ctrl = _controller(tmp_path, mic_busy=lambda: busy["value"])
    ctrl._mic_wait_timeout = 1.0
    started = time.monotonic()

    def release():
        time.sleep(0.3)
        busy["value"] = False

    import threading
    threading.Thread(target=release, daemon=True).start()
    ctrl._process(np.zeros(16000, dtype=np.float32))
    assert time.monotonic() - started >= 0.25


def test_queue_drops_backlog_to_stay_realtime(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    for _ in range(10):
        ctrl._on_segment(np.zeros(1600, dtype=np.float32))
    assert ctrl._queue.qsize() <= ctrl.MAX_QUEUE
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_system_captions.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.core.system_captions'`

- [ ] **Step 3: 實作模組**

建立 `app/core/system_captions.py`：

```python
import queue
import threading
import time

from PySide6.QtCore import QObject, Signal

from .local_translate import LocalTranslator
from .system_audio import SystemAudioCapture


class SystemCaptionsController(QObject):
    """把「系統聲音 → 文字 → 母語翻譯」串起來。

    擷取與處理都在自己的執行緒，不佔用麥克風流程的 executor。
    麥克風正在使用時先讓路，避免使用者說話被系統字幕的辨識卡住。
    """

    caption_ready = Signal(str, str)      # 原文, 母語翻譯
    state_changed = Signal(str, str)      # state, message
    error_occurred = Signal(str)

    MAX_QUEUE = 3  # 積壓超過就丟掉最舊的，即時性優先

    def __init__(self, config, stt, mic_busy, parent=None):
        super().__init__(parent)
        self.config = config
        self.stt = stt
        self._mic_busy = mic_busy
        self._mic_wait_timeout = 10.0
        self._translator = LocalTranslator(
            engine=config.get("system_captions", "engine",
                              default="nllb-600m"),
            compute_device=config.get("system_captions", "compute_device",
                                      default="auto"))
        self._capture = None
        self._queue = queue.Queue()
        self._worker = None
        self._running = False

    # ---- 生命週期 ----

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._running = True
        self._translator.set_engine(
            self.config.get("system_captions", "engine", default="nllb-600m"))
        self._translator.set_compute_device(
            self.config.get("system_captions", "compute_device",
                            default="auto"))
        self._worker = threading.Thread(
            target=self._work, daemon=True, name="system-captions")
        self._worker.start()
        self._capture = SystemAudioCapture(
            device_name=self.config.get("system_captions", "device",
                                        default="default"),
            on_segment=self._on_segment,
            on_error=self._on_capture_error,
            silence_ms=self.config.get("system_captions", "segment_silence_ms",
                                       default=600),
            max_seconds=self.config.get("system_captions", "max_segment_sec",
                                        default=8))
        self._capture.start()
        self.state_changed.emit("listening", "正在聽系統聲音…")

    def stop(self):
        self._running = False
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        self._queue.put(None)  # 叫醒工作執行緒
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(timeout=2.0)
        self.state_changed.emit("idle", "已停止系統聲音字幕")

    # ---- 語言 ----

    def _languages(self):
        """回傳 (辨識語言, 翻譯目標語言)。"""
        native = self.config.get("language", "source", default="zh")
        target = self.config.get("language", "target", default="en")
        spoken = self.config.get("system_captions", "language", default="")
        return (spoken or target), native

    # ---- 管線 ----

    def _on_segment(self, audio):
        while self._queue.qsize() >= self.MAX_QUEUE:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(audio)

    def _on_capture_error(self, error):
        self._running = False
        self.error_occurred.emit(f"系統聲音擷取失敗：{error}")
        self.state_changed.emit("error", "系統聲音擷取失敗")

    def _work(self):
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
        while self._running:
            audio = self._queue.get()
            if audio is None or not self._running:
                break
            try:
                self._process(audio)
            except Exception as e:
                self.error_occurred.emit(f"系統字幕處理失敗：{e}")

    def _process(self, audio):
        # 麥克風優先：使用者正在說話時先讓路
        deadline = time.monotonic() + self._mic_wait_timeout
        while self._mic_busy() and time.monotonic() < deadline:
            time.sleep(0.05)

        spoken, native = self._languages()
        text = (self.stt.transcribe(audio, language=spoken) or "").strip()
        if not text:
            return
        translated = self._translator.translate(text, spoken, native)
        self.caption_ready.emit(text, translated)
```

- [ ] **Step 4: 執行測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS，45 passed

- [ ] **Step 5: Commit**

```bash
git add app/core/system_captions.py tests/test_system_captions.py
git commit -m "feat: 新增系統聲音字幕編排器"
```

---

### Task 6: 浮層共用基底（重構）

**Files:**
- Create: `app/ui/overlay_base.py`
- Modify: `app/ui/subtitle.py`（改為繼承基底、刪除重複的拖動/縮放程式碼）

**Interfaces:**
- Consumes: `Config`
- Produces: `DraggableResizableOverlay(config, flags, parent=None)`
  - 類別屬性：`CONFIG_SECTION: str`（子類必填）、`BORDER = 8`、`DRAG_THRESHOLD = 6`
  - 方法：`saved_pos() -> QPoint | None`、`saved_size() -> tuple[int, int] | None`、`save_geometry()`、`target_opacity() -> float`、`apply_opacity()`、`min_overlay_size() -> QSize`（子類可覆寫）
  - 覆寫點：`_on_simple_click(pos)`（沒有拖動的單擊，預設不做事）、`_on_geometry_changed()`（拖動或縮放結束，預設呼叫 `save_geometry()`）

- [ ] **Step 1: 建立基底模組**

建立 `app/ui/overlay_base.py`：

```python
from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtWidgets import QApplication, QWidget

_MAX = 16777215


class DraggableResizableOverlay(QWidget):
    """無邊框浮層的共用行為：拖動移動、邊框/角落縮放、位置大小記憶、透明度。

    子類別必須設定 CONFIG_SECTION（config.json 中的區段名），
    位置與大小會存進該區段的 pos_x / pos_y / width / height。
    """

    CONFIG_SECTION = ""
    BORDER = 8            # 邊框拖曳調整大小的感應寬度
    DRAG_THRESHOLD = 6

    def __init__(self, config, flags, parent=None):
        super().__init__(parent, flags)
        self.config = config
        self.setMouseTracking(True)
        self._pressed = False
        self._moved = False
        self._resize_edges = None
        self._press_geo = None
        self._press_global = QPoint()
        self._drag_offset = QPoint()

    # ---- 幾何記憶 ----

    def min_overlay_size(self) -> QSize:
        return QSize(240, 100)

    def saved_pos(self):
        x = self.config.get(self.CONFIG_SECTION, "pos_x", default=None)
        y = self.config.get(self.CONFIG_SECTION, "pos_y", default=None)
        if x is None or y is None:
            return None
        pos = QPoint(int(x), int(y))
        # 儲存的位置必須還在某個螢幕上（拔掉外接螢幕後回到預設位置）
        if QApplication.screenAt(pos) is None:
            return None
        return pos

    def saved_size(self):
        w = self.config.get(self.CONFIG_SECTION, "width", default=None)
        h = self.config.get(self.CONFIG_SECTION, "height", default=None)
        if not w or not h:
            return None
        return int(w), int(h)

    def save_geometry(self):
        self.config.set(self.CONFIG_SECTION, "pos_x", self.pos().x())
        self.config.set(self.CONFIG_SECTION, "pos_y", self.pos().y())
        self.config.set(self.CONFIG_SECTION, "width", self.width())
        self.config.set(self.CONFIG_SECTION, "height", self.height())

    def target_opacity(self) -> float:
        percent = self.config.get(self.CONFIG_SECTION, "opacity", default=100)
        return max(0.3, min(1.0, int(percent) / 100))

    def apply_opacity(self):
        if self.isVisible():
            self.setWindowOpacity(self.target_opacity())

    # ---- 覆寫點 ----

    def _on_simple_click(self, pos):
        """單擊（沒有拖動）時呼叫。"""

    def _on_geometry_changed(self):
        """拖動或縮放結束時呼叫。"""
        self.save_geometry()

    # ---- 滑鼠 ----

    def _edges_at(self, pos):
        left = pos.x() <= self.BORDER
        right = pos.x() >= self.width() - self.BORDER
        top = pos.y() <= self.BORDER
        bottom = pos.y() >= self.height() - self.BORDER
        if left or right or top or bottom:
            return (left, right, top, bottom)
        return None

    def _cursor_for(self, edges):
        left, right, top, bottom = edges
        if (left and top) or (right and bottom):
            return Qt.CursorShape.SizeFDiagCursor
        if (left and bottom) or (right and top):
            return Qt.CursorShape.SizeBDiagCursor
        if left or right:
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._pressed = True
        self._moved = False
        self._press_global = event.globalPosition().toPoint()
        self._press_geo = self.geometry()
        self._resize_edges = self._edges_at(event.position().toPoint())
        self._drag_offset = self._press_global - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        gpos = event.globalPosition().toPoint()
        if not self._pressed:
            edges = self._edges_at(pos)
            self.setCursor(self._cursor_for(edges) if edges
                           else Qt.CursorShape.ArrowCursor)
            return
        if not self._moved:
            if (gpos - self._press_global).manhattanLength() < self.DRAG_THRESHOLD:
                return
            self._moved = True
        if self._resize_edges:
            self._apply_resize(gpos)
        else:
            self.move(gpos - self._drag_offset)

    def _apply_resize(self, gpos):
        left, right, top, bottom = self._resize_edges
        geo = self._press_geo
        dx = gpos.x() - self._press_global.x()
        dy = gpos.y() - self._press_global.y()
        min_size = self.minimumSize()
        x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()
        if right:
            w = max(min_size.width(), geo.width() + dx)
        if bottom:
            h = max(min_size.height(), geo.height() + dy)
        if left:
            w = max(min_size.width(), geo.width() - dx)
            x = geo.x() + geo.width() - w
        if top:
            h = max(min_size.height(), geo.height() - dy)
            y = geo.y() + geo.height() - h
        self.setGeometry(x, y, w, h)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self._pressed:
            return
        self._pressed = False
        self._resize_edges = None
        if self._moved:
            self._on_geometry_changed()
        else:
            self._on_simple_click(event.position().toPoint())
```

- [ ] **Step 2: 讓 `SubtitleOverlay` 改用基底**

修改 `app/ui/subtitle.py`：

1. 匯入基底並改變繼承（`class SubtitleOverlay(DraggableResizableOverlay):`）：

```python
from .overlay_base import DraggableResizableOverlay
```

2. `__init__` 開頭改為呼叫基底建構式並設定區段：

```python
class SubtitleOverlay(DraggableResizableOverlay):
    CONFIG_SECTION = "subtitle"

    def __init__(self, config):
        super().__init__(config, _FLAGS)
```
（刪除原本的 `self.config = config`、`setMouseTracking`、`_pressed`/`_moved`/
`_resize_edges`/`_press_geo`/`_press_global`/`_drag_offset` 初始化，這些基底已有。）

3. 刪除 `subtitle.py` 中這些已移到基底的方法：`_edges_at`、`_cursor_for`、
   `_apply_resize`，以及 `mousePressEvent`/`mouseMoveEvent`/`mouseReleaseEvent`
   的通用部分。改為只保留字幕特有的行為，用覆寫點實作：

```python
    def min_overlay_size(self) -> QSize:
        """最小尺寸隨字體連動：至少放得下中英各一行與邊距。"""
        zh, en = self._font_sizes()
        min_w = max(240, en * 6) + 28 + 16 + 102
        min_h = 20 + 18 + int((zh + en) * 1.6) + 6
        return QSize(min_w, min_h)

    def _on_simple_click(self, pos):
        # 編輯中點卡片其他地方 → 視同完成編輯（重新翻譯）
        if self._editing:
            self._finish_edit()
            return
        # 點在原文行上 → 進入編輯；點其他地方 → 關閉
        if self.zh_label.isVisible() and self.zh_label.geometry().contains(pos):
            self._begin_edit()
        else:
            self.dismiss()

    def _on_geometry_changed(self):
        self.save_geometry()
        self._hide_timer.start(self._duration_ms())
```

4. 原本 `mousePressEvent` 中「按到邊框時停止倒數」的行為，改寫成覆寫：

```python
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        # 開始操作（拖動或縮放）就先停止倒數，放開時由 _on_geometry_changed 重啟
        self._hide_timer.stop()
```

5. `_saved_pos()` 的呼叫改用基底的 `saved_pos()`；`_popup()` 中讀取
   `subtitle.width/height` 的程式碼改用 `saved_size()`。

- [ ] **Step 3: 執行既有測試確認沒改壞**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS，45 passed

- [ ] **Step 4: 離屏驗證字幕互動全部正常**

Run:
```bash
.venv/Scripts/python.exe -X utf8 -c "import os; os.environ['QT_QPA_PLATFORM']='offscreen'; from PySide6.QtWidgets import QApplication; from PySide6.QtCore import QTimer, QEventLoop, QPoint, QPointF, QEvent, Qt; from PySide6.QtGui import QMouseEvent; app=QApplication([]); wait=lambda ms:(lambda l:(QTimer.singleShot(ms,l.quit), l.exec()))(QEventLoop()); from app.config import Config; from app.ui.subtitle import SubtitleOverlay; cfg=Config(); s=SubtitleOverlay(cfg); s.show_result('今天天氣很好','Nice weather.', None); wait(300); w0=s.width(); g=s.mapToGlobal(QPoint(s.width()-2, s.height()//2)); pe=QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(s.width()-2, s.height()//2), QPointF(g.x(),g.y()), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier); s.mousePressEvent(pe); mv=QMouseEvent(QEvent.Type.MouseMove, QPointF(0,0), QPointF(g.x()+80,g.y()), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier); s.mouseMoveEvent(mv); re=QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(0,0), QPointF(g.x()+80,g.y()), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier); s.mouseReleaseEvent(re); print('拖右邊框加寬:', s.width()==w0+80); print('大小已記憶:', cfg.get('subtitle','width')==s.width())"
```
Expected: `拖右邊框加寬: True`、`大小已記憶: True`

- [ ] **Step 5: Commit**

```bash
git add app/ui/overlay_base.py app/ui/subtitle.py
git commit -m "refactor: 抽出浮層共用的拖動與縮放基底"
```

---

### Task 7: 系統聲音字幕浮層

**Files:**
- Create: `app/ui/system_subtitle.py`

**Interfaces:**
- Consumes: `DraggableResizableOverlay`（Task 6）、`Config`
- Produces: `SystemSubtitleOverlay(config)`
  - 訊號 `closed_by_user()`
  - 方法 `show_overlay(screen=None)`、`update_caption(original, translated)`、`clear_history()`、`apply_style()`
  - 類別屬性 `CONFIG_SECTION = "system_captions"`

- [ ] **Step 1: 實作浮層**

建立 `app/ui/system_subtitle.py`：

```python
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
)

from qfluentwidgets import FluentIcon, Theme, TransparentToolButton

from .overlay_base import DraggableResizableOverlay

_FLAGS = (Qt.WindowType.FramelessWindowHint
          | Qt.WindowType.WindowStaysOnTopHint
          | Qt.WindowType.Tool
          | Qt.WindowType.WindowDoesNotAcceptFocus)

DEFAULT_BG = "#0d2b33"
ACCENT = "#4dd0e1"      # 青色，與麥克風字幕（深灰白字）明顯區分
MIN_SIZE = QSize(360, 130)


class SystemSubtitleOverlay(DraggableResizableOverlay):
    """系統聲音的雙語字幕：上行原文、下行母語翻譯。

    與麥克風字幕的差異：青藍配色、左上角「🔊 系統聲音」標籤、
    不會自動倒數消失（停止擷取才收起）、可展開本次逐字稿。
    """

    CONFIG_SECTION = "system_captions"
    closed_by_user = Signal()

    def __init__(self, config):
        super().__init__(config, _FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMinimumSize(MIN_SIZE)
        self._history = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 10, 12, 16)
        outer.setSpacing(6)

        header = QHBoxLayout()
        self.tag_label = QLabel("🔊 系統聲音", self)
        self.tag_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 12px; font-weight: bold;"
            "font-family: 'Microsoft JhengHei';")
        header.addWidget(self.tag_label)
        header.addStretch(1)
        self.history_button = TransparentToolButton(
            FluentIcon.HISTORY.icon(Theme.DARK), self)
        self.history_button.setToolTip("展開／收起這次的逐字稿")
        self.history_button.setFixedSize(22, 22)
        self.history_button.clicked.connect(self._toggle_history)
        header.addWidget(self.history_button)
        self.close_button = TransparentToolButton(
            FluentIcon.CLOSE.icon(Theme.DARK), self)
        self.close_button.setToolTip("關閉系統聲音字幕")
        self.close_button.setFixedSize(22, 22)
        self.close_button.clicked.connect(self._on_close)
        header.addWidget(self.close_button)
        outer.addLayout(header)

        self.original_label = QLabel("", self)
        self.original_label.setWordWrap(True)
        outer.addWidget(self.original_label)

        self.translated_label = QLabel("", self)
        self.translated_label.setWordWrap(True)
        outer.addWidget(self.translated_label)
        outer.addStretch(1)

        self.history_view = QTextEdit(self)
        self.history_view.setReadOnly(True)
        self.history_view.hide()
        self.history_view.setStyleSheet(
            "QTextEdit { background: rgba(255,255,255,18); color: white;"
            " border: 1px solid rgba(255,255,255,45); border-radius: 6px;"
            " font-size: 13px; font-family: 'Microsoft JhengHei'; }")
        outer.addWidget(self.history_view)

        self.apply_style()

    # ---- 對外 API ----

    def show_overlay(self, screen=None):
        pos = self.saved_pos()
        if pos is not None:
            screen = QApplication.screenAt(pos)
        screen = screen or QApplication.primaryScreen()
        geo = screen.availableGeometry()
        size = self.saved_size()
        if size:
            width = max(size[0], MIN_SIZE.width())
            height = max(size[1], MIN_SIZE.height())
        else:
            width = max(min(int(geo.width() * 0.5), 780), MIN_SIZE.width())
            height = MIN_SIZE.height()
        self.resize(width, height)
        if pos is not None:
            self.move(pos)
        else:
            # 預設放上方，與麥克風字幕（預設下方）天然分開
            self.move(geo.center().x() - width // 2, geo.top() + 64)
        self.apply_style()
        self.show()
        self.setWindowOpacity(self.target_opacity())

    def update_caption(self, original: str, translated: str):
        self.original_label.setText(original)
        self.translated_label.setText(translated)
        self._history.append((original, translated))
        if self.history_view.isVisible():
            self._refresh_history()

    def clear_history(self):
        self._history = []
        self.history_view.clear()

    def apply_style(self):
        size = int(self.config.get(
            self.CONFIG_SECTION, "font_size", default=20))
        original_font = QFont("Segoe UI")
        original_font.setPixelSize(max(12, round(size * 0.8)))
        self.original_label.setFont(original_font)
        self.original_label.setStyleSheet("color: #cfe9ef;")
        translated_font = QFont("Microsoft JhengHei")
        translated_font.setPixelSize(size)
        translated_font.setBold(True)
        self.translated_label.setFont(translated_font)
        self.translated_label.setStyleSheet("color: white;")
        self.update()

    # ---- 內部 ----

    def _toggle_history(self):
        if self.history_view.isVisible():
            self.history_view.hide()
        else:
            self._refresh_history()
            self.history_view.show()

    def _refresh_history(self):
        lines = []
        for original, translated in self._history:
            lines.append(f"{original}\n{translated}\n")
        self.history_view.setPlainText("\n".join(lines))
        self.history_view.moveCursor(self.history_view.textCursor()
                                     .MoveOperation.End)

    def _on_close(self):
        self.hide()
        self.closed_by_user.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor(self.config.get(
            self.CONFIG_SECTION, "bg_color", default=DEFAULT_BG))
        bg.setAlpha(225)
        painter.setBrush(bg)
        painter.setPen(QColor(ACCENT))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 14, 14)
```

- [ ] **Step 2: 離屏驗證顯示與歷史**

Run:
```bash
.venv/Scripts/python.exe -X utf8 -c "import os; os.environ['QT_QPA_PLATFORM']='offscreen'; from PySide6.QtWidgets import QApplication; from PySide6.QtCore import QTimer, QEventLoop; app=QApplication([]); wait=lambda ms:(lambda l:(QTimer.singleShot(ms,l.quit), l.exec()))(QEventLoop()); from app.config import Config; from app.ui.system_subtitle import SystemSubtitleOverlay; cfg=Config(); o=SystemSubtitleOverlay(cfg); o.show_overlay(); wait(200); o.update_caption('The meeting starts now.','會議現在開始。'); o.update_caption('Please mute your microphone.','請把麥克風靜音。'); wait(200); print('顯示中:', o.isVisible()); print('原文:', o.original_label.text()); print('譯文:', o.translated_label.text()); o.history_button.click(); wait(100); print('歷史可見:', o.history_view.isVisible()); print('歷史含兩句:', o.history_view.toPlainText().count('會議')+o.history_view.toPlainText().count('靜音')==2)"
```
Expected: `顯示中: True`、原文/譯文為最後一句、`歷史可見: True`、`歷史含兩句: True`

- [ ] **Step 3: Commit**

```bash
git add app/ui/system_subtitle.py
git commit -m "feat: 新增系統聲音雙語字幕浮層"
```

---

### Task 8: 重疊推開幾何

**Files:**
- Modify: `app/ui/overlay_base.py`（在檔案末尾加入模組層級函式）
- Test: `tests/test_overlay_geometry.py`

**Interfaces:**
- Consumes: 無（純幾何）
- Produces: `push_away(mover: QRect, fixed: QRect, bounds: QRect, margin: int = 12) -> QPoint`
  回傳 `mover` 應移到的新左上角座標；不重疊時原樣回傳 `mover.topLeft()`

- [ ] **Step 1: 寫失敗測試**

建立 `tests/test_overlay_geometry.py`：

```python
import sys
from pathlib import Path

from PySide6.QtCore import QRect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ui.overlay_base import push_away

BOUNDS = QRect(0, 0, 1920, 1080)


def test_no_overlap_keeps_position():
    mover = QRect(100, 100, 400, 100)
    fixed = QRect(100, 800, 400, 100)
    assert push_away(mover, fixed, BOUNDS) == mover.topLeft()


def test_overlap_pushes_up_when_room_above():
    mover = QRect(100, 500, 400, 100)
    fixed = QRect(100, 520, 400, 100)
    new_pos = push_away(mover, fixed, BOUNDS)
    moved = QRect(new_pos, mover.size())
    assert not moved.intersects(fixed.adjusted(-12, -12, 12, 12))
    assert moved.bottom() <= fixed.top()


def test_overlap_pushes_down_when_no_room_above():
    mover = QRect(100, 10, 400, 100)
    fixed = QRect(100, 20, 400, 100)
    new_pos = push_away(mover, fixed, BOUNDS)
    moved = QRect(new_pos, mover.size())
    assert not moved.intersects(fixed.adjusted(-12, -12, 12, 12))
    assert moved.top() >= fixed.bottom()


def test_result_stays_inside_bounds():
    mover = QRect(100, 1000, 400, 100)
    fixed = QRect(100, 990, 400, 100)
    new_pos = push_away(mover, fixed, BOUNDS)
    moved = QRect(new_pos, mover.size())
    assert BOUNDS.contains(moved)


def test_horizontally_separated_boxes_are_untouched():
    mover = QRect(0, 500, 400, 100)
    fixed = QRect(1000, 500, 400, 100)
    assert push_away(mover, fixed, BOUNDS) == mover.topLeft()
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_overlay_geometry.py -q`
Expected: FAIL，`ImportError: cannot import name 'push_away'`

- [ ] **Step 3: 實作函式**

在 `app/ui/overlay_base.py` 末尾加入：

```python
def push_away(mover, fixed, bounds, margin: int = 12):
    """算出 mover 要移到哪裡才不會和 fixed 重疊（只上下移動）。

    mover / fixed / bounds 都是 QRect；回傳新的左上角 QPoint。
    不重疊就原樣回傳。優先往移動距離較短、且放得下的方向讓開。
    """
    from PySide6.QtCore import QPoint, QRect

    padded = fixed.adjusted(-margin, -margin, margin, margin)
    if not mover.intersects(padded):
        return mover.topLeft()

    up_y = fixed.top() - margin - mover.height()
    down_y = fixed.bottom() + margin
    candidates = []
    if up_y >= bounds.top():
        candidates.append((abs(up_y - mover.top()), up_y))
    if down_y + mover.height() <= bounds.bottom():
        candidates.append((abs(down_y - mover.top()), down_y))
    if not candidates:
        # 兩邊都放不下：貼著邊界，至少不要跑出畫面
        y = max(bounds.top(),
                min(down_y, bounds.bottom() - mover.height()))
        return QPoint(mover.left(), y)
    candidates.sort()
    return QPoint(mover.left(), candidates[0][1])
```

- [ ] **Step 4: 執行測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS，50 passed

- [ ] **Step 5: Commit**

```bash
git add app/ui/overlay_base.py tests/test_overlay_geometry.py
git commit -m "feat: 新增字幕重疊推開的幾何計算"
```

---

### Task 9: 設定頁 UI

**Files:**
- Modify: `app/ui/settings_page.py`

**Interfaces:**
- Consumes: `ENGINE_LABELS`（Task 3）、`SystemAudioCapture.list_output_devices`（Task 2）、`LANGUAGES`（既有）
- Produces:
  - `SettingsInterface.system_captions_switch`（`SwitchButton`）
  - `SettingsInterface.set_system_captions_checked(checked: bool)`
  - 訊號 `system_captions_toggled(bool)`、`system_captions_settings_changed()`

- [ ] **Step 1: 新增設定區塊**

在 `app/ui/settings_page.py` 匯入：

```python
from ..core.local_translate import ENGINE_LABELS
from ..core.system_audio import SystemAudioCapture
```

在訊號區加入：

```python
    system_captions_toggled = Signal(bool)
    system_captions_settings_changed = Signal()
```

在「文法檢查」區塊之後、「懸浮球字幕」區塊之前插入 UI：

```python
        # ---- 系統聲音字幕 ----
        layout.addSpacing(8)
        layout.addWidget(StrongBodyLabel("系統聲音字幕", self.view))
        self.system_captions_switch = SwitchButton(self.view)
        self._add_row(layout, "啟用（把電腦播放的聲音即時翻成母語字幕）",
                      self.system_captions_switch)
        self.system_hotkey_button = PushButton("", self.view)
        self.system_hotkey_button.setToolTip("點一下，然後按任意按鍵或滑鼠鍵")
        self._add_row(layout, "系統字幕開關熱鍵", self.system_hotkey_button)
        self.system_device_combo = ComboBox(self.view)
        self.system_device_combo.addItem("系統預設")
        try:
            self.system_device_combo.addItems(
                SystemAudioCapture.list_output_devices())
        except Exception:
            pass
        self._add_row(layout, "擷取來源（哪個喇叭的聲音）",
                      self.system_device_combo)
        self.system_language_combo = ComboBox(self.view)
        self.system_language_combo.addItem("跟隨目標語言")
        self.system_language_combo.addItems([n for _, n in LANGUAGES])
        self._add_row(layout, "系統聲音的語言", self.system_language_combo)
        self.system_engine_combo = ComboBox(self.view)
        self.system_engine_combo.addItems([n for _, n in ENGINE_LABELS])
        self._add_row(layout, "本機翻譯模型", self.system_engine_combo)
        self.system_compute_combo = ComboBox(self.view)
        self.system_compute_combo.addItems(["自動（優先 GPU）", "只用 CPU"])
        self._add_row(layout, "翻譯運算裝置", self.system_compute_combo)
        self.system_font_spin = SpinBox(self.view)
        self.system_font_spin.setRange(12, 48)
        self.system_font_spin.setSuffix(" px")
        self._add_row(layout, "系統字幕字體大小", self.system_font_spin)
        self.system_opacity_spin = SpinBox(self.view)
        self.system_opacity_spin.setRange(30, 100)
        self.system_opacity_spin.setSingleStep(5)
        self.system_opacity_spin.setSuffix(" %")
        self._add_row(layout, "系統字幕透明度", self.system_opacity_spin)
```

- [ ] **Step 2: 載入設定值**

在 `_load_from_config` 末尾加入：

```python
        sc = "system_captions"
        self.system_captions_switch.setChecked(
            cfg.get(sc, "enabled", default=False))
        self.system_device_combo.setCurrentIndex(0)
        device = cfg.get(sc, "device", default="default")
        if device != "default":
            for i in range(1, self.system_device_combo.count()):
                if self.system_device_combo.itemText(i) == device:
                    self.system_device_combo.setCurrentIndex(i)
                    break
        self.system_language_combo.setCurrentIndex(0)
        spoken = cfg.get(sc, "language", default="")
        if spoken:
            codes = [c for c, _ in LANGUAGES]
            if spoken in codes:
                self.system_language_combo.setCurrentIndex(
                    codes.index(spoken) + 1)
        engine = cfg.get(sc, "engine", default="nllb-600m")
        engine_ids = [e for e, _ in ENGINE_LABELS]
        if engine in engine_ids:
            self.system_engine_combo.setCurrentIndex(engine_ids.index(engine))
        self.system_compute_combo.setCurrentIndex(
            1 if cfg.get(sc, "compute_device", default="auto") == "cpu" else 0)
        self.system_font_spin.setValue(cfg.get(sc, "font_size", default=20))
        self.system_opacity_spin.setValue(cfg.get(sc, "opacity", default=100))
```

在 `_refresh_hotkey_button` 末尾加入：

```python
        self.system_hotkey_button.setText(hotkey_display(
            self.config.get("system_captions", "hotkey_type",
                            default="keyboard"),
            self.config.get("system_captions", "hotkey_key", default="f11")))
```

- [ ] **Step 3: 連接訊號**

在 `_connect_signals` 加入：

```python
        self.system_captions_switch.checkedChanged.connect(
            self._on_system_captions_switch)
        self.system_hotkey_button.clicked.connect(
            self._on_system_hotkey_button)
        self.system_device_combo.currentIndexChanged.connect(
            self._on_system_settings_changed)
        self.system_language_combo.currentIndexChanged.connect(
            self._on_system_settings_changed)
        self.system_engine_combo.currentIndexChanged.connect(
            self._on_system_settings_changed)
        self.system_compute_combo.currentIndexChanged.connect(
            self._on_system_settings_changed)
        self.system_font_spin.valueChanged.connect(
            self._on_system_settings_changed)
        self.system_opacity_spin.valueChanged.connect(
            self._on_system_settings_changed)
```

新增方法：

```python
    def _on_system_captions_switch(self, checked: bool):
        if self._loading:
            return
        # config 寫入由 MainWindow.set_system_captions_enabled 統一處理
        self.system_captions_toggled.emit(checked)

    def set_system_captions_checked(self, checked: bool):
        """由主視窗回寫（熱鍵或字幕 ✕ 改了狀態時同步 UI）。"""
        self._loading = True
        self.system_captions_switch.setChecked(checked)
        self._loading = False

    def _on_system_settings_changed(self, _value=None):
        if self._loading:
            return
        sc = "system_captions"
        index = self.system_device_combo.currentIndex()
        self.config.set(sc, "device", "default" if index == 0
                        else self.system_device_combo.currentText())
        lang_index = self.system_language_combo.currentIndex()
        self.config.set(sc, "language", "" if lang_index == 0
                        else LANGUAGES[lang_index - 1][0])
        self.config.set(sc, "engine",
                        ENGINE_LABELS[self.system_engine_combo.currentIndex()][0])
        self.config.set(sc, "compute_device",
                        "cpu" if self.system_compute_combo.currentIndex() == 1
                        else "auto")
        self.config.set(sc, "font_size", self.system_font_spin.value())
        self.config.set(sc, "opacity", self.system_opacity_spin.value())
        self.system_captions_settings_changed.emit()

    def _on_system_hotkey_button(self):
        self.hotkey_capture_started.emit()
        dialog = HotkeyCaptureDialog(self.window())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            taken = self._existing_hotkeys() | {
                (self.config.get("grammar", "hotkey_type", default="keyboard"),
                 self.config.get("grammar", "hotkey_key", default="f10"))}
            if (dialog.result_type, dialog.result_key) in taken:
                self.error_requested.emit("系統字幕熱鍵不能跟其他熱鍵相同")
            else:
                self.config.set("system_captions", "hotkey_type",
                                dialog.result_type)
                self.config.set("system_captions", "hotkey_key",
                                dialog.result_key)
                self._refresh_hotkey_button()
        self.hotkey_changed.emit()
```

- [ ] **Step 4: 主視窗連接設定頁訊號**

在 `app/ui/main_window.py` 的 `MainWindow.__init__` 加入：

```python
        self.settings.system_captions_toggled.connect(
            self.set_system_captions_enabled)
        self.settings.system_captions_settings_changed.connect(
            self._apply_system_caption_style)
```

新增方法：

```python
    def _apply_system_caption_style(self):
        self.system_subtitle.apply_style()
        self.system_subtitle.apply_opacity()
```

- [ ] **Step 5: 執行測試與離屏驗證**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS，50 passed

接著執行 Task 10 Step 4 的離屏驗證指令。
Expected: 全部為 True

- [ ] **Step 6: Commit**

```bash
git add app/ui/settings_page.py app/ui/main_window.py
git commit -m "feat: 系統聲音字幕設定頁"
```

---

### Task 10: 主視窗接線（熱鍵、選單、開關同步、推開）

**Files:**
- Modify: `app/controller.py`（新增熱鍵監聽與 `mic_busy`）
- Modify: `app/ui/bubble.py`（長按選單新增項目）
- Modify: `app/ui/main_window.py`（建立元件、三處同步、推開、結束收拾）

**Interfaces:**
- Consumes: `SystemCaptionsController`（Task 5）、`SystemSubtitleOverlay`（Task 7）、`push_away`（Task 8）
- Produces:
  - `AppController.system_hotkey`（`HotkeyListener`）、`AppController.apply_system_hotkey()`、訊號 `system_hotkey_pressed`
  - `AppController.mic_busy() -> bool`
  - `BubbleWidget.system_captions_toggle_requested`（訊號）
  - `MainWindow.set_system_captions_enabled(enabled: bool)`

- [ ] **Step 1: `AppController` 新增熱鍵與忙碌查詢**

在 `app/controller.py` 的訊號區加入：

```python
    system_hotkey_pressed = Signal()          # 系統字幕開關熱鍵
```

在 `__init__` 建立監聽器（`self.grammar_hotkey` 之後）：

```python
        self.system_hotkey = HotkeyListener(
            self.system_hotkey_pressed.emit, lambda: None)
```

在 `start()` 中加入 `self.apply_system_hotkey()`；在 `shutdown()` 中加入
`self.system_hotkey.stop()`。新增方法：

```python
    def apply_system_hotkey(self):
        enabled = self.config.get("system_captions", "enabled", default=False)
        hotkey_type = self.config.get(
            "system_captions", "hotkey_type", default="keyboard")
        key_name = self.config.get(
            "system_captions", "hotkey_key", default="f11")
        # 熱鍵永遠掛著（否則關掉後就沒辦法用熱鍵開回來）
        if not key_name:
            self.system_hotkey.stop()
            return
        try:
            self.system_hotkey.configure(hotkey_type, key_name)
        except ValueError as e:
            self.error_occurred.emit(str(e))

    def mic_busy(self) -> bool:
        """麥克風正在錄音或處理中——系統字幕要讓路。"""
        return self._session_active
```

- [ ] **Step 2: 懸浮球選單新增項目**

在 `app/ui/bubble.py` 的 `BubbleMenu` 新增訊號與按鈕：

```python
    system_captions_clicked = Signal()
```

把按鈕清單改為：

```python
        for text, signal in (
                ("⌨  輸入框 開/關", self.input_clicked),
                ("🔊  系統字幕 開/關", self.system_captions_clicked),
                ("🗖  開啟主視窗", self.window_clicked),
                ("✕  結束程式", self.quit_clicked)):
```

在 `BubbleWidget` 新增轉發訊號：

```python
    system_captions_toggle_requested = Signal()
```

並在 `__init__` 連接：

```python
        self.menu.system_captions_clicked.connect(
            self.system_captions_toggle_requested)
```

- [ ] **Step 3: 主視窗建立元件並接線**

在 `app/ui/main_window.py` 匯入：

```python
from ..core.system_captions import SystemCaptionsController
from .overlay_base import push_away
from .system_subtitle import SystemSubtitleOverlay
```

在 `MainWindow.__init__` 中（`self.float_input` 建立之後）加入：

```python
        self.system_captions = SystemCaptionsController(
            config, controller.stt, controller.mic_busy, self)
        self.system_subtitle = SystemSubtitleOverlay(config)
        self.system_captions.caption_ready.connect(self._on_system_caption)
        self.system_captions.state_changed.connect(self.home.set_state)
        self.system_captions.error_occurred.connect(self._show_error)
        self.system_captions.error_occurred.connect(
            lambda _msg: self.set_system_captions_enabled(False))
        self.system_subtitle.closed_by_user.connect(
            lambda: self.set_system_captions_enabled(False))
        self.bubble.system_captions_toggle_requested.connect(
            self._toggle_system_captions)
        controller.system_hotkey_pressed.connect(self._toggle_system_captions)
```

新增方法：

```python
    def _toggle_system_captions(self):
        self.set_system_captions_enabled(
            not self.config.get("system_captions", "enabled", default=False))

    def set_system_captions_enabled(self, enabled: bool):
        """熱鍵、設定頁開關、字幕 ✕ 三處共用的同一個狀態。"""
        self.config.set("system_captions", "enabled", enabled)
        self.settings.set_system_captions_checked(enabled)
        if enabled:
            self.system_subtitle.clear_history()
            self.system_subtitle.show_overlay(self._current_screen())
            self._avoid_overlap(self.system_subtitle, self.subtitle)
            self.system_captions.start()
        else:
            self.system_captions.stop()
            self.system_subtitle.hide()

    def _current_screen(self):
        return self.screen() if self.isVisible() else self.bubble.screen()

    def _on_system_caption(self, original: str, translated: str):
        self.system_subtitle.update_caption(original, translated)
        self._avoid_overlap(self.system_subtitle, self.subtitle)

    def _avoid_overlap(self, mover, fixed):
        """兩個字幕都在畫面上時，把 mover 推開到不重疊的位置。"""
        if not mover.isVisible() or not fixed.isVisible():
            return
        screen = (QApplication.screenAt(mover.frameGeometry().center())
                  or QApplication.primaryScreen())
        target = push_away(mover.frameGeometry(), fixed.frameGeometry(),
                           screen.availableGeometry())
        if target != mover.pos():
            mover.move(target)
            mover.save_geometry()
```

在 `_on_result_ready` 中，麥克風字幕顯示後也要避讓（加在字幕顯示之後）：

```python
            self._avoid_overlap(self.subtitle, self.system_subtitle)
```

在 `_on_hotkey_changed` 與 `_on_settings_reset` 中加入
`self.controller.apply_system_hotkey()`。

在 `_quit()` 中加入：

```python
        self.system_captions.stop()
        self.system_subtitle.hide()
```

在 `settings.hotkey_capture_started` 的連接串中加入
`self.settings.hotkey_capture_started.connect(controller.system_hotkey.stop)`。

- [ ] **Step 4: 離屏驗證開關同步與推開**

Run:
```bash
.venv/Scripts/python.exe -X utf8 -c "import os; os.environ['QT_QPA_PLATFORM']='offscreen'; from PySide6.QtWidgets import QApplication; from PySide6.QtCore import QTimer, QEventLoop; app=QApplication([]); wait=lambda ms:(lambda l:(QTimer.singleShot(ms,l.quit), l.exec()))(QEventLoop()); from app.config import Config; from app.controller import AppController; from app.ui.main_window import MainWindow; cfg=Config(); c=AppController(cfg); w=MainWindow(cfg,c); w.show(); wait(50); w.system_captions.start=lambda: None; w.system_captions.stop=lambda: None; w._toggle_system_captions(); wait(200); print('開啟後字幕可見:', w.system_subtitle.isVisible(), '| config:', cfg.get('system_captions','enabled'), '| 設定開關:', w.settings.system_captions_switch.isChecked()); w.subtitle.show_result('中文','English', None); wait(200); w._avoid_overlap(w.subtitle, w.system_subtitle); wait(100); print('兩字幕不重疊:', not w.subtitle.frameGeometry().intersects(w.system_subtitle.frameGeometry())); w._toggle_system_captions(); wait(200); print('關閉後隱藏:', not w.system_subtitle.isVisible(), '| config:', cfg.get('system_captions','enabled'))"
```
Expected: `開啟後字幕可見: True`、config `True`、設定開關 `True`、`兩字幕不重疊: True`、`關閉後隱藏: True`、config `False`

（`system_captions_switch` 與 `set_system_captions_checked` 已於 Task 9 建立。）

- [ ] **Step 5: Commit**

```bash
git add app/controller.py app/ui/bubble.py app/ui/main_window.py
git commit -m "feat: 系統字幕接線（熱鍵、懸浮球選單、開關同步、重疊推開）"
```

---

### Task 11: 端到端驗證與文件

**Files:**
- Modify: `README.md`
- Modify: `config.example.json`（由 `DEFAULT_CONFIG` 重新產生）

**Interfaces:**
- Consumes: 前面所有任務
- Produces: 可交付的功能與最新文件

- [ ] **Step 1: 重新產生範例設定檔**

Run:
```bash
.venv/Scripts/python.exe -c "import copy, json; from app.config import DEFAULT_CONFIG; d=copy.deepcopy(DEFAULT_CONFIG); d['ai']['deepseek']['api_key']='在這裡填入你的 DeepSeek API key'; json.dump(d, open('config.example.json','w',encoding='utf-8'), ensure_ascii=False, indent=2); print('done')"
```
Expected: `done`，且 `config.example.json` 含 `system_captions` 區段

- [ ] **Step 2: 端到端實測（人工）**

1. 啟動程式：`Start-Process "D:\Code\AI\AI英文溝通\AI語音中翻英.exe"`
2. 設定頁開啟「系統聲音字幕」，確認狀態列出現「正在聽系統聲音…」
3. 播放一段英文 YouTube 影片
4. 確認：畫面上方出現青藍色字幕框，上行英文原文、下行中文翻譯，內容持續更新
5. 按住 F9 說中文 → 麥克風字幕出現在下方，兩個字幕不重疊
6. 點系統字幕的歷史按鈕 → 展開本次逐字稿，可選取複製
7. 按 F11 → 系統字幕停止並消失；再按 F11 → 重新開始
8. 拖動系統字幕到麥克風字幕位置 → 放開後其中一個被推開

Expected: 上述每一項都成立；字幕延遲約在講完一句後 1 秒內出現

- [ ] **Step 3: 更新 README**

在 `README.md` 的「使用方式」新增一節：

```markdown
### 系統聲音字幕（聽電腦在播什麼）
- 設定頁「系統聲音字幕」開啟（或按 F11、懸浮球長按選單）後，程式會擷取
  **電腦正在播放的聲音**（YouTube、線上會議…，不含你的麥克風），
  即時辨識並用**本機模型**翻成母語，以青藍色雙語字幕顯示在畫面上方。
- 翻譯不走 DeepSeek，模型可在設定頁切換（NLLB 600M / 1.3B / OPUS-MT），
  首次使用會自動下載，之後離線可用。
- 字幕可拖動、調大小、調透明度；與麥克風字幕重疊時會自動推開。
- 「歷史」按鈕可展開本次的整段逐字稿並複製。
```

在「架構重點」新增：

```markdown
- 系統聲音擷取（`app/core/system_audio.py`）：PortAudio 沒有 loopback 裝置，
  改用 `soundcard` 的 WASAPI loopback；能量門檻切句後才送辨識。
- 本機翻譯（`app/core/local_translate.py`）：直接用 faster-whisper already
  安裝的 CTranslate2 跑 NLLB/OPUS-MT，刻意避開 Argos Translate
  （會引入 torch+spacy+stanza 約 2–3GB）。
```

- [ ] **Step 4: 全套測試與 lint 檢查**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS，50 passed

Run: `.venv/Scripts/python.exe -c "import app.main, app.ui.main_window, app.ui.system_subtitle, app.core.system_captions; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 5: Commit**

```bash
git add README.md config.example.json
git commit -m "docs: 補上系統聲音字幕的使用說明與範例設定"
```

---

## 自我檢查（撰寫者已完成）

- **規格覆蓋**：spec 的每個元件（system_audio / local_translate /
  system_captions / overlay_base / system_subtitle）、設定區段、三處開關、
  重疊推開、錯誤處理（擷取失敗自動關閉、GPU 退回 CPU、空辨識略過、
  麥克風優先）、測試策略，均對應到 Task 1–11。
- **無 placeholder**：所有步驟都有可直接執行的指令或完整程式碼。
- **型別一致**：`push_away` 回傳 `QPoint`；`caption_ready(str, str)` 與
  `update_caption(original, translated)` 參數順序一致；`repo_for` 回傳
  `(repo, kind)` 與 `ensure_loaded` 的解構一致；`set_system_captions_checked` 於 Task 9（設定頁）定義、Task 10（接線）使用，順序正確。
