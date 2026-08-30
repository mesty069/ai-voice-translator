# 串流式三行系統字幕 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把系統聲音字幕改成 Live Captions 風格：每秒重新辨識「上一句句尾之後」的音訊，目前句邊講邊長、文字穩定就翻、變了就重翻；疊加層固定顯示最近 3 行（可設 1–5），舊句往上頂掉。

**Architecture:** `SystemAudioCapture` 改成只吐連續音框進 `RollingAudioBuffer`；新模組 `streaming_captions.py` 放純邏輯（`split_sentences_by_words`、`CaptionState`）和執行緒引擎 `StreamingCaptionEngine`；`SystemCaptionsController` 只負責組裝與轉 Qt 訊號；`SystemSubtitleOverlay.set_rows()` 畫多行。faster-whisper 開 `word_timestamps=True`，用句尾字的時間戳推進音訊起點。

**Tech Stack:** Python 3.12、numpy、faster-whisper（CTranslate2）、PySide6 + qfluentwidgets、pytest。Spec：`docs/superpowers/specs/2026-08-30-streaming-captions-design.md`。

## Global Constraints

- 常數與值（放 `app/core/streaming_captions.py`，一字不差）：`PUNC_EOS = ".?!。？！"`、`SHORT_THRESHOLD = 10`、`MEDIUM_THRESHOLD = 40`、`WINDOW_MAX_SEC = 12.0`、`POLL_SEC = 1.0`、`IDLE_ROUNDS = 2`、`SILENCE_COMMIT_SEC = 1.5`、`DISPLAY_ROWS = 3`、`MIN_PEAK = 0.01`
- config `system_captions.display_rows` 預設 `3`，設定頁範圍 1–5，變更即時生效（不重啟管線）；移除 `segment_silence_ms`、`max_segment_sec` 兩鍵（config、example、設定頁、spec 之外的文件）
- 辨識一律 `beam_size=1`；`SpeechToText.transcribe()`（麥克風用）行為不得改變
- 不改麥克風翻譯管線；不引入 torch 或任何新依賴；系統字幕不呼叫 DeepSeek
- 所有使用者可見文字為繁體中文；commit 訊息繁體中文、`feat:`/`fix:`/`refactor:`/`test:` 前綴
- 世代編號機制保留：`stop()` 之後舊 engine／capture 的任何回呼不得再發訊號；GUI 執行緒等待任何背景執行緒不得超過 0.2 秒
- 測試命令：`.venv\Scripts\python.exe -m pytest tests -q`；每個任務結束時全綠
- 絕不 `git add -A` / `git add .`；`config.json` 不得進 git

---

### Task 1: RollingAudioBuffer 與只吐音框的 SystemAudioCapture

**Files:**
- Modify: `app/core/system_audio.py`（刪 `SegmentAccumulator`，新增 `RollingAudioBuffer`，`SystemAudioCapture` 改回呼）
- Rewrite: `tests/test_system_audio.py`

**Interfaces:**
- Produces:
  - `RollingAudioBuffer(sample_rate=16000, max_seconds=60.0)`；`append(frames: np.ndarray) -> None`；`total_seconds -> float`（自建立以來收到的總秒數，絕對時間軸）；`start_seconds -> float`（目前保留區起點的絕對秒）；`since(t_sec: float) -> np.ndarray`（絕對秒 t 到尾端，t 早於保留區就從保留區起點）；`trim_before(t_sec: float) -> None`；`tail_peak(seconds: float) -> float`（尾端 seconds 秒內 `|x|` 最大值，沒資料回 0.0）
  - `SystemAudioCapture(device_name="default", on_frames=None, on_error=None)`；`on_frames(np.ndarray)` 每 0.1 秒一塊 float32 單聲道；`start()/stop()/is_running/list_output_devices()` 不變

- [ ] **Step 1: 改寫測試檔（先寫失敗測試）**

把 `tests/test_system_audio.py` 整個換成：

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.system_audio import (  # noqa: E402
    SAMPLE_RATE,
    RollingAudioBuffer,
    SystemAudioCapture,
)


def _sec(n, value=0.5):
    return np.full(int(SAMPLE_RATE * n), value, dtype=np.float32)


def test_append_advances_absolute_time():
    buf = RollingAudioBuffer()
    buf.append(_sec(1.0))
    buf.append(_sec(0.5))
    assert abs(buf.total_seconds - 1.5) < 1e-6
    assert buf.start_seconds == 0.0


def test_since_returns_audio_from_absolute_time():
    buf = RollingAudioBuffer()
    buf.append(_sec(1.0, 0.1))
    buf.append(_sec(1.0, 0.2))
    out = buf.since(1.0)
    assert len(out) == SAMPLE_RATE
    assert np.allclose(out, 0.2)


def test_trim_keeps_absolute_time_axis():
    buf = RollingAudioBuffer()
    buf.append(_sec(2.0, 0.1))
    buf.trim_before(1.5)
    buf.append(_sec(1.0, 0.3))
    assert abs(buf.start_seconds - 1.5) < 1e-6
    assert abs(buf.total_seconds - 3.0) < 1e-6
    out = buf.since(2.0)          # 絕對秒 2.0 起 = 剛加進去的那 1 秒
    assert len(out) == SAMPLE_RATE
    assert np.allclose(out, 0.3)


def test_since_before_start_clamps_to_start():
    buf = RollingAudioBuffer()
    buf.append(_sec(2.0, 0.1))
    buf.trim_before(1.0)
    assert len(buf.since(0.0)) == SAMPLE_RATE


def test_max_seconds_auto_trims():
    buf = RollingAudioBuffer(max_seconds=2.0)
    buf.append(_sec(3.0))
    assert buf.total_seconds - buf.start_seconds <= 2.0 + 1e-6
    assert abs(buf.total_seconds - 3.0) < 1e-6


def test_tail_peak():
    buf = RollingAudioBuffer()
    buf.append(_sec(1.0, 0.8))
    buf.append(_sec(1.0, 0.0))
    assert buf.tail_peak(0.5) == 0.0
    assert buf.tail_peak(1.5) > 0.7
    assert RollingAudioBuffer().tail_peak(1.0) == 0.0


def test_since_with_empty_buffer_is_empty():
    assert len(RollingAudioBuffer().since(0.0)) == 0


# ---- SystemAudioCapture：假 soundcard，確認每塊音框都原樣送出 ----

class _FakeRecorder:
    def __init__(self, blocks):
        self._blocks = list(blocks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def record(self, numframes):
        if self._blocks:
            return self._blocks.pop(0)
        import time
        time.sleep(0.01)
        return np.zeros((numframes, 1), dtype=np.float32)


class _FakeMic:
    def __init__(self, blocks):
        self._blocks = blocks

    def recorder(self, samplerate, channels):
        return _FakeRecorder(self._blocks)


class _FakeSpeaker:
    name = "Fake Speaker"


class _FakeSoundcard:
    def __init__(self, blocks):
        self._blocks = blocks

    def default_speaker(self):
        return _FakeSpeaker()

    def all_speakers(self):
        return [_FakeSpeaker()]

    def get_microphone(self, name, include_loopback=False):
        return _FakeMic(self._blocks)


def test_capture_forwards_every_block(monkeypatch):
    import time
    blocks = [np.full((1600, 1), 0.1, dtype=np.float32),
              np.full((1600, 1), 0.2, dtype=np.float32)]
    monkeypatch.setitem(sys.modules, "soundcard", _FakeSoundcard(blocks))
    got = []
    cap = SystemAudioCapture(on_frames=lambda f: got.append(f.copy()))
    cap.start()
    deadline = time.monotonic() + 2.0
    while len(got) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    cap.stop()
    assert len(got) >= 2
    assert got[0].ndim == 1 and len(got[0]) == 1600
    assert np.allclose(got[0], 0.1) and np.allclose(got[1], 0.2)


def test_capture_error_is_reported(monkeypatch):
    import time

    class _Broken(_FakeSoundcard):
        def get_microphone(self, name, include_loopback=False):
            raise RuntimeError("no loopback")

    monkeypatch.setitem(sys.modules, "soundcard", _Broken([]))
    errors = []
    cap = SystemAudioCapture(on_error=errors.append)
    cap.start()
    deadline = time.monotonic() + 2.0
    while not errors and time.monotonic() < deadline:
        time.sleep(0.01)
    assert errors and "no loopback" in str(errors[0])
    assert cap.is_running is False
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_system_audio.py -q`
Expected: FAIL，`ImportError: cannot import name 'RollingAudioBuffer'`

- [ ] **Step 3: 實作**

把 `app/core/system_audio.py` 整個換成：

```python
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
                    if self._on_frames is not None:
                        self._on_frames(np.asarray(frames, dtype=np.float32))
        except Exception as e:
            self._running = False
            if self._on_error is not None:
                self._on_error(e)
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_system_audio.py -q`
Expected: 9 passed

注意：此時 `tests/test_system_captions.py` 會因 `SystemAudioCapture` 簽名改變而部分失敗——那是 Task 5 的範圍，本任務只要求 `test_system_audio.py` 全綠。**在本任務的 commit 訊息註明「test_system_captions 於 Task 5 重寫」。**

- [ ] **Step 5: Commit**

```bash
git add app/core/system_audio.py tests/test_system_audio.py
git commit -m "refactor: 系統聲音擷取改為連續音框 + RollingAudioBuffer（串流字幕基礎；test_system_captions 於後續任務重寫）"
```

---

### Task 2: SpeechToText.transcribe_words（帶時間戳）

**Files:**
- Modify: `app/core/stt.py`
- Test: `tests/test_stt_words.py`（新）

**Interfaces:**
- Produces: `Word = namedtuple("Word", "text start end")`（模組層級，`from app.core.stt import Word`）；`SpeechToText.transcribe_words(audio: np.ndarray, language: str = "en", beam_size: int = 1) -> list[Word]`：`text` 已 `strip()`、空字略過；`start/end` 為秒、相對於 `audio` 起點；在 `self._lock` 內執行；模型未載入 `raise RuntimeError("語音模型尚未載入完成")`（與 `transcribe` 同訊息）。

- [ ] **Step 1: 寫失敗測試**

建立 `tests/test_stt_words.py`：

```python
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.stt import SpeechToText, Word  # noqa: E402


class _FakeModel:
    def __init__(self, delay=0.0):
        self.calls = []
        self.delay = delay

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        time.sleep(self.delay)
        seg1 = SimpleNamespace(text=" Hello world.", words=[
            SimpleNamespace(word=" Hello", start=0.1, end=0.4),
            SimpleNamespace(word=" world.", start=0.5, end=0.9),
        ])
        seg2 = SimpleNamespace(text=" Next", words=[
            SimpleNamespace(word=" Next", start=1.2, end=1.5),
            SimpleNamespace(word="  ", start=1.5, end=1.6),   # 空字要略過
        ])
        return iter([seg1, seg2]), None


def _stt_with(model):
    stt = SpeechToText()
    stt._model = model
    return stt


def test_transcribe_words_returns_stripped_words_with_times():
    stt = _stt_with(_FakeModel())
    words = stt.transcribe_words(np.zeros(16000, dtype=np.float32), "en")
    assert words == [Word("Hello", 0.1, 0.4), Word("world.", 0.5, 0.9),
                     Word("Next", 1.2, 1.5)]


def test_transcribe_words_requests_word_timestamps_and_greedy():
    model = _FakeModel()
    stt = _stt_with(model)
    stt.transcribe_words(np.zeros(16000, dtype=np.float32), "en")
    kwargs = model.calls[0]
    assert kwargs["word_timestamps"] is True
    assert kwargs["beam_size"] == 1
    assert kwargs["language"] == "en"


def test_transcribe_words_without_model_raises():
    stt = SpeechToText()
    with pytest.raises(RuntimeError, match="尚未載入"):
        stt.transcribe_words(np.zeros(16000, dtype=np.float32), "en")


def test_transcribe_words_shares_lock_with_transcribe():
    """麥克風的 transcribe 與字幕的 transcribe_words 不得同時進 GPU。"""
    model = _FakeModel(delay=0.05)
    stt = _stt_with(model)
    active, max_active = [0], [0]
    counter_lock = threading.Lock()
    original = model.transcribe

    def counted(audio, **kwargs):
        with counter_lock:
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
        try:
            return original(audio, **kwargs)
        finally:
            with counter_lock:
                active[0] -= 1

    model.transcribe = counted
    audio = np.zeros(16000, dtype=np.float32)
    threads = [threading.Thread(target=lambda: stt.transcribe(audio, "en")),
               threading.Thread(target=lambda: stt.transcribe_words(audio, "en"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(2.0)
    assert max_active[0] == 1
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stt_words.py -q`
Expected: FAIL，`ImportError: cannot import name 'Word'`

- [ ] **Step 3: 實作**

在 `app/core/stt.py` 頂端 `import threading` 之後加：

```python
from collections import namedtuple
```

在 `class SpeechToText:` 之前加：

```python
Word = namedtuple("Word", "text start end")  # text 已 strip；秒，相對於送進去的音訊起點
```

在 `transcribe()` 方法之後加：

```python
    def transcribe_words(self, audio: np.ndarray, language: str = "en",
                         beam_size: int = 1) -> list:
        """串流字幕用：回傳帶時間戳的字。句尾字的 end 用來推進音訊起點。

        與 transcribe() 共用同一把鎖（同一顆模型、同一張 GPU）。
        """
        with self._lock:
            model = self._model
            if model is None:
                raise RuntimeError("語音模型尚未載入完成")
            segments, _info = model.transcribe(
                audio,
                language=language,
                beam_size=beam_size,
                vad_filter=True,
                word_timestamps=True,
                initial_prompt="以下是繁體中文的句子。" if language == "zh" else None,
            )
            words = []
            for seg in segments:
                for w in (getattr(seg, "words", None) or []):
                    text = (w.word or "").strip()
                    if text:
                        words.append(Word(text, float(w.start), float(w.end)))
            return words
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stt_words.py tests/test_stt_lock.py -q`
Expected: 全部 passed

- [ ] **Step 5: Commit**

```bash
git add app/core/stt.py tests/test_stt_words.py
git commit -m "feat: SpeechToText.transcribe_words 回傳帶時間戳的字（串流字幕用）"
```

---

### Task 3: 純邏輯——句子切分與 CaptionState

**Files:**
- Create: `app/core/streaming_captions.py`
- Test: `tests/test_streaming_state.py`（新）

**Interfaces:**
- Consumes: `Word(text, start, end)` from `app.core.stt`
- Produces（全部在 `app/core/streaming_captions.py`）:
  - 常數：`PUNC_EOS`, `SHORT_THRESHOLD`, `MEDIUM_THRESHOLD`, `WINDOW_MAX_SEC`, `POLL_SEC`, `IDLE_ROUNDS`, `SILENCE_COMMIT_SEC`, `DISPLAY_ROWS`, `MIN_PEAK`（值見 Global Constraints）
  - `@dataclass Sentence: text: str; end: float`
  - `@dataclass Row: original: str; translated: str = ""; is_final: bool = False`
  - `split_sentences_by_words(words) -> tuple[list[Sentence], Sentence | None]`：以 `PUNC_EOS` 結尾的字收尾一句；字之間用空白接（`join_words`）；最後沒收尾的字組成 `current`
  - `join_words(texts: list[str]) -> str`：CJK 字之間不加空白，其餘加一個空白
  - `class CaptionState(display_rows=DISPLAY_ROWS)`：`set_display_rows(n)`、`rows -> list[Row]`（`finals + [current]` 取最後 `display_rows` 個，回傳副本）、`update_current(text) -> bool`（回傳「這一輪要不要翻譯 current」）、`set_current_translation(translated)`、`commit_text(text) -> Row`（完成句進 finals，current 清空）、`commit_current() -> Row | None`（把 current 直接當完成句，保留其翻譯）、`translate_input(text) -> str`（短句接前一句原文）、`previous_final_text -> str`

- [ ] **Step 1: 寫失敗測試**

建立 `tests/test_streaming_state.py`：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.stt import Word  # noqa: E402
from app.core.streaming_captions import (  # noqa: E402
    IDLE_ROUNDS,
    MEDIUM_THRESHOLD,
    SHORT_THRESHOLD,
    CaptionState,
    Row,
    Sentence,
    join_words,
    split_sentences_by_words,
)


def _w(text, start, end):
    return Word(text, start, end)


# ---- join_words ----

def test_join_words_latin_uses_spaces():
    assert join_words(["Hello", "world."]) == "Hello world."


def test_join_words_cjk_has_no_spaces():
    assert join_words(["你好", "世界。"]) == "你好世界。"


def test_join_words_mixed():
    assert join_words(["我用", "Python", "寫程式。"]) == "我用 Python 寫程式。"


# ---- split_sentences_by_words ----

def test_split_completed_and_current():
    words = [_w("The", 0.0, 0.2), _w("end.", 0.3, 0.6),
             _w("Next", 0.9, 1.1), _w("one", 1.2, 1.4)]
    completed, current = split_sentences_by_words(words)
    assert completed == [Sentence("The end.", 0.6)]
    assert current == Sentence("Next one", 1.4)


def test_split_all_complete_has_no_current():
    completed, current = split_sentences_by_words(
        [_w("Hi.", 0.0, 0.3), _w("Bye!", 0.5, 0.8)])
    assert [s.text for s in completed] == ["Hi.", "Bye!"]
    assert current is None


def test_split_empty():
    assert split_sentences_by_words([]) == ([], None)


def test_split_cjk_eos():
    completed, current = split_sentences_by_words(
        [_w("你好", 0.0, 0.4), _w("。", 0.4, 0.5), _w("再", 0.7, 0.9)])
    assert completed == [Sentence("你好。", 0.5)]
    assert current == Sentence("再", 0.9)


# ---- CaptionState ----

def test_current_grows_and_translates_after_idle_rounds():
    st = CaptionState(display_rows=3)
    assert st.update_current("The meeting") is False
    assert st.rows == [Row("The meeting", "", False)]
    assert st.update_current("The meeting will") is False   # 變了、不夠長
    for _ in range(IDLE_ROUNDS - 1):
        assert st.update_current("The meeting will") is False
    assert st.update_current("The meeting will") is True     # idle 達門檻
    st.set_current_translation("會議將")
    assert st.rows == [Row("The meeting will", "會議將", False)]


def test_same_text_is_not_retranslated():
    st = CaptionState()
    for _ in range(IDLE_ROUNDS + 1):
        need = st.update_current("Stable text")
    assert need is True
    st.set_current_translation("穩定")
    for _ in range(5):
        assert st.update_current("Stable text") is False


def test_long_change_translates_immediately():
    st = CaptionState()
    long_text = "x" * MEDIUM_THRESHOLD
    assert st.update_current(long_text) is True


def test_eos_translates_immediately():
    st = CaptionState()
    assert st.update_current("Done.") is True


def test_translation_kept_while_text_grows():
    st = CaptionState()
    st.update_current("Hello")
    st.set_current_translation("你好")
    st.update_current("Hello there")
    assert st.rows[-1] == Row("Hello there", "你好", False)


def test_commit_text_pushes_rows_and_caps_display():
    st = CaptionState(display_rows=3)
    st.update_current("partial")
    r1 = st.commit_text("One.")
    r1.translated = "一。"
    assert st.rows == [Row("One.", "一。", True)]     # current 清掉了
    st.commit_text("Two.")
    st.commit_text("Three.")
    st.commit_text("Four.")
    st.update_current("Five")
    assert [r.original for r in st.rows] == ["Three.", "Four.", "Five"]
    st.set_display_rows(2)
    assert [r.original for r in st.rows] == ["Four.", "Five"]


def test_commit_current_keeps_translation():
    st = CaptionState()
    st.update_current("Trailing")
    st.set_current_translation("尾句")
    row = st.commit_current()
    assert row == Row("Trailing", "尾句", True)
    assert st.rows == [row]
    assert st.commit_current() is None


def test_translate_input_prepends_previous_for_short_text():
    st = CaptionState()
    st.commit_text("Previous sentence here.")
    short = "x" * (SHORT_THRESHOLD - 1)
    assert st.translate_input(short) == "Previous sentence here. " + short
    assert st.translate_input("x" * SHORT_THRESHOLD) == "x" * SHORT_THRESHOLD


def test_translate_input_without_previous_is_unchanged():
    assert CaptionState().translate_input("ok") == "ok"


def test_rows_returns_copies():
    st = CaptionState()
    st.update_current("a")
    st.rows[0].original = "mutated"
    assert st.rows[0].original == "a"
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_streaming_state.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.core.streaming_captions'`

- [ ] **Step 3: 實作**

建立 `app/core/streaming_captions.py`：

```python
"""串流式系統字幕的純邏輯（本檔上半）與引擎（Task 4 補在下半）。

做法照 SakiRinn/LiveCaptions-Translator：只翻「最後一個句尾標點之後」的
目前句，文字穩定或夠長就翻、相同不重翻；完成句進歷史、疊加層顯示最近 N 行。
"""
from dataclasses import dataclass, replace

PUNC_EOS = ".?!。？！"
SHORT_THRESHOLD = 10       # 目前句短於此 → 翻譯時接前一句
MEDIUM_THRESHOLD = 40      # 文字變了且長於此 → 立即重翻
WINDOW_MAX_SEC = 12.0      # 開放音訊窗上限
POLL_SEC = 1.0             # 兩輪辨識的最小間隔
IDLE_ROUNDS = 2            # 文字連續幾輪沒變就翻
SILENCE_COMMIT_SEC = 1.5   # 尾端靜音多久把未完句視為完成
DISPLAY_ROWS = 3           # 預設顯示行數
MIN_PEAK = 0.01            # 低於此視為靜音


@dataclass
class Sentence:
    text: str
    end: float   # 最後一個字的 end（秒，相對於送進辨識的音訊起點）


@dataclass
class Row:
    original: str
    translated: str = ""
    is_final: bool = False


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF
            or 0xFF00 <= code <= 0xFFEF or 0x3000 <= code <= 0x303F)


def join_words(texts: list) -> str:
    """CJK 字之間不加空白，其餘字之間加一個空白。"""
    out = ""
    for text in texts:
        if not text:
            continue
        if out and not (_is_cjk(out[-1]) or _is_cjk(text[0])):
            out += " "
        out += text
    return out


def split_sentences_by_words(words):
    """把帶時間戳的字切成 (完成句列表, 目前未完句或 None)。"""
    completed, pending = [], []
    for w in words:
        pending.append(w)
        if w.text and w.text[-1] in PUNC_EOS:
            completed.append(Sentence(join_words([p.text for p in pending]),
                                      pending[-1].end))
            pending = []
    current = None
    if pending:
        current = Sentence(join_words([p.text for p in pending]), pending[-1].end)
    return completed, current


class CaptionState:
    """rows 與「要不要翻譯」的決策。無執行緒、無 IO，方便測試。"""

    def __init__(self, display_rows=DISPLAY_ROWS):
        self.display_rows = display_rows
        self._finals = []
        self._current = None
        self._idle = 0
        self._last_translated = None   # 上次送翻譯的 current 原文

    def set_display_rows(self, n: int):
        self.display_rows = max(1, int(n))

    @property
    def rows(self) -> list:
        rows = list(self._finals)
        if self._current is not None:
            rows.append(self._current)
        return [replace(r) for r in rows[-self.display_rows:]]

    @property
    def previous_final_text(self) -> str:
        return self._finals[-1].original if self._finals else ""

    def update_current(self, text: str) -> bool:
        """更新目前句原文，回傳這一輪是否要翻譯它。"""
        if self._current is None:
            self._current = Row(text)
            changed = True
            self._idle = 0
        elif text != self._current.original:
            self._current.original = text   # 翻譯先留著，重翻後才換
            changed = True
            self._idle = 0
        else:
            changed = False
            self._idle += 1
        if not text or text == self._last_translated:
            return False
        ends_sentence = text[-1] in PUNC_EOS
        return (ends_sentence or self._idle >= IDLE_ROUNDS
                or (changed and len(text) >= MEDIUM_THRESHOLD))

    def set_current_translation(self, translated: str):
        if self._current is not None:
            self._current.translated = translated
            self._last_translated = self._current.original

    def commit_text(self, text: str) -> Row:
        """辨識器已判定完成的句子：進 finals，目前句清空。"""
        row = Row(text, "", True)
        self._finals.append(row)
        self._reset_current()
        return row

    def commit_current(self):
        """把目前句直接當完成句（靜音／窗超長），保留已有翻譯。"""
        if self._current is None:
            return None
        row = self._current
        row.is_final = True
        self._finals.append(row)
        self._reset_current()
        return row

    def translate_input(self, text: str) -> str:
        prev = self.previous_final_text
        if prev and len(text) < SHORT_THRESHOLD:
            return f"{prev} {text}"
        return text

    def _reset_current(self):
        self._current = None
        self._idle = 0
        self._last_translated = None
        # 只保留顯示所需 + 一點餘裕，歷史另由 UI 記錄
        del self._finals[:-max(self.display_rows, 5)]
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_streaming_state.py -q`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add app/core/streaming_captions.py tests/test_streaming_state.py
git commit -m "feat: 串流字幕純邏輯——依時間戳切句、CaptionState 決定何時翻譯與顯示幾行"
```

---

### Task 4: StreamingCaptionEngine（每秒重新辨識開放音訊窗）

**Files:**
- Modify: `app/core/streaming_captions.py`（追加引擎）
- Test: `tests/test_streaming_engine.py`（新）

**Interfaces:**
- Consumes: `RollingAudioBuffer`（Task 1）、`stt.transcribe_words`/`stt.is_ready`（Task 2）、`CaptionState`/`split_sentences_by_words`（Task 3）、`translator.translate(text, src, tgt, progress_cb=None)`、`ModelLoadError` from `app.core.local_translate`
- Produces: `StreamingCaptionEngine(buffer, stt, translator, languages, mic_busy, on_rows, on_state, on_fatal, on_final, display_rows=DISPLAY_ROWS, now=time.monotonic, sleep=time.sleep)`
  - `languages() -> (spoken, native)`；`on_rows(list[Row])`；`on_state(state: str, msg: str)`；`on_fatal(msg: str)`；`on_final(original: str, translated: str)`（完成句拿到翻譯時，給歷史面板）
  - `start()` / `stop()`（stop 只把 `_running=False` 並 join 0.1 秒）；`step()`（跑一輪，可直接測）；`set_display_rows(n)`；`committed_t: float`；`STT_WAIT_TIMEOUT = 300.0`
  - `step()` 回傳 `True` 表示有跑辨識（測試用）

- [ ] **Step 1: 寫失敗測試**

建立 `tests/test_streaming_engine.py`：

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.local_translate import ModelLoadError  # noqa: E402
from app.core.stt import Word  # noqa: E402
from app.core.streaming_captions import (  # noqa: E402
    IDLE_ROUNDS,
    SILENCE_COMMIT_SEC,
    WINDOW_MAX_SEC,
    StreamingCaptionEngine,
)
from app.core.system_audio import SAMPLE_RATE, RollingAudioBuffer  # noqa: E402


class _ScriptedStt:
    """每次呼叫依序回傳劇本裡的 words；用完就重複最後一組。"""

    def __init__(self, script, ready=True):
        self.script = list(script)
        self.calls = []
        self.is_ready = ready

    def transcribe_words(self, audio, language="en", beam_size=1):
        self.calls.append((len(audio), language, beam_size))
        if len(self.script) > 1:
            return self.script.pop(0)
        return self.script[0] if self.script else []


class _FakeTranslator:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def translate(self, text, src, tgt, progress_cb=None):
        if self.fail is not None:
            raise self.fail
        self.calls.append(text)
        return f"譯[{text}]"


class _Sink:
    def __init__(self):
        self.rows, self.states, self.fatals, self.finals = [], [], [], []


def _engine(stt, translator=None, buffer=None, sink=None, mic_busy=lambda: False):
    sink = sink or _Sink()
    buffer = buffer or RollingAudioBuffer()
    eng = StreamingCaptionEngine(
        buffer, stt, translator or _FakeTranslator(),
        languages=lambda: ("en", "zh"), mic_busy=mic_busy,
        on_rows=lambda rows: sink.rows.append(rows),
        on_state=lambda s, m: sink.states.append((s, m)),
        on_fatal=sink.fatals.append,
        on_final=lambda o, t: sink.finals.append((o, t)),
        now=lambda: 0.0, sleep=lambda s: None)
    return eng, buffer, sink


def _speech(seconds, level=0.3):
    return np.full(int(SAMPLE_RATE * seconds), level, dtype=np.float32)


def _silence(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def test_step_skips_when_no_audio():
    eng, buf, sink = _engine(_ScriptedStt([[]]))
    assert eng.step() is False
    assert sink.rows == []


def test_current_sentence_shows_then_translates_when_stable():
    stt = _ScriptedStt([[Word("The", 0.0, 0.2), Word("meeting", 0.3, 0.6)]])
    tr = _FakeTranslator()
    eng, buf, sink = _engine(stt, tr)
    buf.append(_speech(1.0))
    eng.step()
    assert sink.rows[-1][-1].original == "The meeting"
    assert sink.rows[-1][-1].translated == ""
    assert tr.calls == []
    for _ in range(IDLE_ROUNDS):
        buf.append(_speech(1.0))
        eng.step()
    assert tr.calls == ["The meeting"]
    assert sink.rows[-1][-1].translated == "譯[The meeting]"
    assert sink.rows[-1][-1].is_final is False


def test_completed_sentence_commits_and_advances_window():
    words = [Word("Hello", 0.0, 0.3), Word("there.", 0.4, 0.8),
             Word("Next", 1.2, 1.5)]
    stt = _ScriptedStt([words, [Word("Next", 0.0, 0.3)]])
    tr = _FakeTranslator()
    eng, buf, sink = _engine(stt, tr)
    buf.append(_speech(2.0))
    eng.step()
    assert abs(eng.committed_t - 0.8) < 1e-6
    assert tr.calls == ["Hello there."]
    assert sink.finals == [("Hello there.", "譯[Hello there.]")]
    rows = sink.rows[-1]
    assert rows[0] == rows[0].__class__("Hello there.", "譯[Hello there.]", True)
    assert rows[1].original == "Next" and rows[1].is_final is False
    # 下一輪只送 committed_t 之後的音訊
    eng.step()
    assert stt.calls[-1][0] == int(SAMPLE_RATE * 2.0) - int(SAMPLE_RATE * 0.8)
    assert abs(buf.start_seconds - 0.8) < 1e-6   # 已 trim


def test_short_final_sentence_is_translated_with_previous():
    words1 = [Word("A", 0.0, 0.1), Word("fairly", 0.2, 0.4), Word("long", 0.5, 0.7),
              Word("sentence.", 0.8, 1.0)]
    words2 = [Word("Yes.", 0.0, 0.2)]
    stt = _ScriptedStt([words1, words2])
    tr = _FakeTranslator()
    eng, buf, sink = _engine(stt, tr)
    buf.append(_speech(1.5))
    eng.step()
    buf.append(_speech(1.0))
    eng.step()
    assert tr.calls[-1] == "A fairly long sentence. Yes."


def test_trailing_silence_commits_current():
    stt = _ScriptedStt([[Word("Trailing", 0.0, 0.4)]])
    eng, buf, sink = _engine(stt)
    buf.append(_speech(1.0))
    eng.step()
    assert sink.rows[-1][-1].is_final is False
    buf.append(_silence(SILENCE_COMMIT_SEC + 0.1))
    eng.step()
    assert sink.rows[-1][-1].is_final is True
    assert sink.rows[-1][-1].original == "Trailing"
    assert abs(eng.committed_t - buf.total_seconds) < 1e-6


def test_window_overflow_forces_commit():
    stt = _ScriptedStt([[Word("Endless", 0.0, 0.5), Word("talk", 0.6, 1.0)]])
    eng, buf, sink = _engine(stt)
    buf.append(_speech(WINDOW_MAX_SEC + 1.0))
    eng.step()
    assert sink.rows[-1][-1].is_final is True
    assert abs(eng.committed_t - 1.0) < 1e-6   # 推進到最後一個字的 end


def test_mic_busy_skips_round():
    stt = _ScriptedStt([[Word("x", 0.0, 0.1)]])
    eng, buf, sink = _engine(stt, mic_busy=lambda: True)
    buf.append(_speech(1.0))
    assert eng.step() is False
    assert stt.calls == []


def test_transient_error_reports_state_and_continues():
    class _Boom:
        is_ready = True

        def transcribe_words(self, *a, **k):
            raise RuntimeError("gpu hiccup")
    eng, buf, sink = _engine(_Boom())
    buf.append(_speech(1.0))
    eng.step()
    assert sink.fatals == []
    assert sink.states[-1][0] == "error"


def test_model_load_error_is_fatal():
    stt = _ScriptedStt([[Word("Done.", 0.0, 0.3)]])
    eng, buf, sink = _engine(stt, _FakeTranslator(fail=ModelLoadError("no model")))
    buf.append(_speech(1.0))
    eng.step()
    assert sink.fatals == ["no model"]
    assert eng.is_running is False


def test_waits_for_stt_ready_and_reports_loading():
    stt = _ScriptedStt([[Word("x", 0.0, 0.1)]], ready=False)
    eng, buf, sink = _engine(stt)
    buf.append(_speech(1.0))
    assert eng.step() is False
    assert sink.states[-1][0] == "loading"
    assert stt.calls == []


def test_stop_makes_callbacks_inert():
    stt = _ScriptedStt([[Word("x", 0.0, 0.1)]])
    eng, buf, sink = _engine(stt)
    buf.append(_speech(1.0))
    eng.stop()
    eng.step()
    assert sink.rows == []


def test_set_display_rows_applies_next_round():
    stt = _ScriptedStt([[Word("One.", 0.0, 0.2), Word("Two.", 0.3, 0.5),
                         Word("Three.", 0.6, 0.8), Word("Four", 0.9, 1.0)]])
    eng, buf, sink = _engine(stt)
    buf.append(_speech(1.5))
    eng.step()
    assert len(sink.rows[-1]) == 3
    eng.set_display_rows(2)
    buf.append(_speech(0.5))
    eng.step()
    assert len(sink.rows[-1]) == 2
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_streaming_engine.py -q`
Expected: FAIL，`ImportError: cannot import name 'StreamingCaptionEngine'`

- [ ] **Step 3: 實作**

在 `app/core/streaming_captions.py` 頂端 `from dataclasses import ...` 上方加：

```python
import threading
import time

from .local_translate import ModelLoadError
```

檔案結尾追加：

```python
class StreamingCaptionEngine:
    """每 POLL_SEC 一輪：辨識「上一句句尾之後」的音訊 → 切句 → 翻譯 → on_rows。

    所有回呼都在引擎執行緒上呼叫；stop() 之後一律不再回呼。
    now/sleep 可注入以便測試。
    """

    STT_WAIT_TIMEOUT = 300.0
    MIN_AUDIO_SEC = 0.5

    def __init__(self, buffer, stt, translator, languages, mic_busy,
                 on_rows, on_state, on_fatal, on_final,
                 display_rows=DISPLAY_ROWS, now=time.monotonic, sleep=time.sleep):
        self.buffer = buffer
        self.stt = stt
        self.translator = translator
        self._languages = languages
        self._mic_busy = mic_busy
        self._on_rows = on_rows
        self._on_state = on_state
        self._on_fatal = on_fatal
        self._on_final = on_final
        self._now = now
        self._sleep = sleep
        self.state = CaptionState(display_rows)
        self.committed_t = 0.0
        self._running = True
        self._thread = None
        self._stt_wait_started = None
        self._loading_announced = False

    # ---- 生命週期 ----

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="streaming-captions")
        self._thread.start()

    def stop(self):
        self._running = False
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=0.1)

    def set_display_rows(self, n: int):
        self.state.set_display_rows(n)

    def _run(self):
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
        while self._running:
            started = self._now()
            self.step()
            elapsed = self._now() - started
            remaining = POLL_SEC - elapsed
            # 分段睡，stop() 才能在 0.1 秒內收到
            while remaining > 0 and self._running:
                self._sleep(min(0.1, remaining))
                remaining -= 0.1

    # ---- 一輪 ----

    def step(self) -> bool:
        if not self._running:
            return False
        try:
            return self._step()
        except ModelLoadError as e:
            self._running = False
            self._on_fatal(str(e))
            return False
        except Exception as e:
            if self._running:
                self._on_state("error", f"系統字幕處理失敗：{e}")
            return False

    def _step(self) -> bool:
        if self._mic_busy():
            return False   # 麥克風優先：使用者在說話時讓路
        if not self._wait_stt():
            return False
        audio = self.buffer.since(self.committed_t)
        sample_rate = self.buffer.sample_rate
        if len(audio) < sample_rate * self.MIN_AUDIO_SEC:
            return False

        spoken, native = self._languages()
        base_t = self.committed_t
        words = self.stt.transcribe_words(audio, language=spoken, beam_size=1)
        if not self._running:
            return False
        completed, current = split_sentences_by_words(words)
        window_sec = self.buffer.total_seconds - base_t

        # 窗太長又沒有任何句尾：把目前這串強制當一句
        if not completed and current is not None and window_sec > WINDOW_MAX_SEC:
            completed, current = [current], None

        for sentence in completed:
            row = self.state.commit_text(sentence.text)
            self.committed_t = base_t + sentence.end
            translated = self._translate(sentence.text, spoken, native)
            row.translated = translated
            self._on_final(sentence.text, translated)
        if completed:
            self.buffer.trim_before(self.committed_t)

        if current is not None:
            if self.state.update_current(current.text):
                self.state.set_current_translation(
                    self._translate(current.text, spoken, native))
            # 尾端安靜夠久 → 這句講完了
            if self.buffer.tail_peak(SILENCE_COMMIT_SEC) < MIN_PEAK:
                row = self.state.commit_current()
                if row is not None:
                    if not row.translated:
                        row.translated = self._translate(row.original, spoken, native)
                    self._on_final(row.original, row.translated)
                self.committed_t = self.buffer.total_seconds
                self.buffer.trim_before(self.committed_t)
        elif not completed:
            # 這段音訊辨識不出字（雜音、音樂）：太長就丟掉，別一直重算
            if window_sec > WINDOW_MAX_SEC:
                self.committed_t = self.buffer.total_seconds
                self.buffer.trim_before(self.committed_t)

        if self._running:
            self._on_rows(self.state.rows)
        return True

    # ---- 輔助 ----

    def _translate(self, text: str, spoken: str, native: str) -> str:
        announced = []

        def progress(message):
            if self._running:
                announced.append(True)
                self._on_state("loading", message)

        result = self.translator.translate(self.state.translate_input(text),
                                           spoken, native, progress_cb=progress)
        if announced and self._running:
            self._on_state("listening", "正在聽系統聲音…")
        return result

    def _wait_stt(self) -> bool:
        """語音模型還在載入：回報一次「載入中」，這輪跳過；逾時視為致命。"""
        if self.stt.is_ready:
            self._stt_wait_started = None
            self._loading_announced = False
            return True
        now = self._now()
        if self._stt_wait_started is None:
            self._stt_wait_started = now
        if not self._loading_announced:
            self._loading_announced = True
            self._on_state("loading", "語音模型載入中，系統字幕稍後開始…")
        if now - self._stt_wait_started >= self.STT_WAIT_TIMEOUT:
            raise ModelLoadError("語音模型載入逾時，系統字幕已停止")
        return False
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_streaming_engine.py tests/test_streaming_state.py -q`
Expected: 全部 passed（12 + 17）

- [ ] **Step 5: Commit**

```bash
git add app/core/streaming_captions.py tests/test_streaming_engine.py
git commit -m "feat: StreamingCaptionEngine——每秒重新辨識開放音訊窗，句尾推進、靜音收句、穩定才翻"
```

---

### Task 5: SystemCaptionsController 重寫 + config

**Files:**
- Rewrite: `app/core/system_captions.py`
- Modify: `app/config.py`（`system_captions` 區段）、`config.example.json`
- Rewrite: `tests/test_system_captions.py`
- Modify: `tests/test_config.py::test_system_captions_defaults`

**Interfaces:**
- Consumes: Task 1/2/4 全部
- Produces: `SystemCaptionsController(config, stt, mic_busy, parent=None)`；訊號 `rows_changed = Signal(object)`（`list[Row]`）、`sentence_finalized = Signal(str, str)`、`state_changed = Signal(str, str)`、`fatal_error = Signal(str)`；`start()/stop()/is_running`；`set_display_rows(n)`；測試 seam `_capture_factory`、`_engine_factory`

- [ ] **Step 1: config**

`app/config.py` 的 `system_captions` 區段：刪掉

```python
        "segment_silence_ms": 400,
        "max_segment_sec": 4,
```

在 `"opacity"` 那行之後加：

```python
        "display_rows": 3,        # 疊加層同時顯示幾行（最近 N-1 句 + 正在講的這句）
```

`config.example.json`：同樣刪 `segment_silence_ms`/`max_segment_sec`，加 `"display_rows": 3`（注意 JSON 逗號）。

`tests/test_config.py::test_system_captions_defaults` 最後一行改為：

```python
    assert cfg.get("system_captions", "display_rows") == 3
    assert cfg.get("system_captions", "segment_silence_ms") is None
```

- [ ] **Step 2: 改寫測試檔**

`tests/test_system_captions.py` 整個換成：

```python
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config  # noqa: E402
from app.core.streaming_captions import Row  # noqa: E402
from app.core.system_captions import SystemCaptionsController  # noqa: E402


class _FakeStt:
    is_ready = True

    def transcribe_words(self, audio, language="en", beam_size=1):
        return []


class _FakeCapture:
    instances = []

    def __init__(self, device_name="default", on_frames=None, on_error=None):
        self.device_name = device_name
        self.on_frames = on_frames
        self.on_error = on_error
        self.started = False
        self.stopped = False
        _FakeCapture.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _FakeEngine:
    instances = []

    def __init__(self, buffer, stt, translator, languages, mic_busy,
                 on_rows, on_state, on_fatal, on_final, display_rows=3):
        self.buffer = buffer
        self.languages = languages
        self.on_rows, self.on_state = on_rows, on_state
        self.on_fatal, self.on_final = on_fatal, on_final
        self.display_rows = display_rows
        self.started = self.stopped = False
        _FakeEngine.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def set_display_rows(self, n):
        self.display_rows = n


def _controller(tmp_path):
    _FakeCapture.instances.clear()
    _FakeEngine.instances.clear()
    cfg = Config(tmp_path / "config.json")
    cfg.set("language", "source", "zh")
    cfg.set("language", "target", "en")
    cfg.set("system_captions", "display_rows", 4)
    ctrl = SystemCaptionsController(cfg, _FakeStt(), lambda: False)
    ctrl._capture_factory = _FakeCapture
    ctrl._engine_factory = _FakeEngine
    return cfg, ctrl


def test_start_wires_capture_into_buffer_and_engine(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    ctrl.start()
    cap, eng = _FakeCapture.instances[0], _FakeEngine.instances[0]
    assert cap.started and eng.started
    assert eng.display_rows == 4
    assert eng.languages() == ("en", "zh")     # 辨識目標語言、翻成母語
    cap.on_frames(np.zeros(1600, dtype=np.float32))
    assert abs(eng.buffer.total_seconds - 0.1) < 1e-6
    ctrl.stop()
    assert cap.stopped and eng.stopped


def test_explicit_spoken_language_overrides_target(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    cfg.set("system_captions", "language", "ja")
    ctrl.start()
    assert _FakeEngine.instances[0].languages() == ("ja", "zh")
    ctrl.stop()


def test_rows_and_finals_are_forwarded_as_signals(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    got_rows, got_finals = [], []
    ctrl.rows_changed.connect(got_rows.append)
    ctrl.sentence_finalized.connect(lambda o, t: got_finals.append((o, t)))
    ctrl.start()
    eng = _FakeEngine.instances[0]
    eng.on_rows([Row("a", "甲", True)])
    eng.on_final("a", "甲")
    assert got_rows == [[Row("a", "甲", True)]]
    assert got_finals == [("a", "甲")]
    ctrl.stop()


def test_callbacks_from_old_generation_are_ignored(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    got = []
    ctrl.rows_changed.connect(got.append)
    ctrl.start()
    old_eng, old_cap = _FakeEngine.instances[0], _FakeCapture.instances[0]
    ctrl.stop()
    old_eng.on_rows([Row("stale")])
    old_cap.on_frames(np.zeros(1600, dtype=np.float32))
    assert got == []
    ctrl.start()
    new_eng = _FakeEngine.instances[1]
    assert new_eng.buffer.total_seconds == 0.0     # 舊音框沒漏進新緩衝
    old_eng.on_rows([Row("stale again")])
    assert got == []
    ctrl.stop()


def test_capture_error_is_fatal(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    fatals = []
    ctrl.fatal_error.connect(fatals.append)
    ctrl.start()
    _FakeCapture.instances[0].on_error(RuntimeError("dead"))
    assert fatals and "dead" in fatals[0]
    assert ctrl.is_running is False


def test_engine_fatal_is_forwarded(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    fatals = []
    ctrl.fatal_error.connect(fatals.append)
    ctrl.start()
    _FakeEngine.instances[0].on_fatal("no model")
    assert fatals == ["no model"]
    assert ctrl.is_running is False


def test_set_display_rows_reaches_running_engine(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    ctrl.start()
    ctrl.set_display_rows(2)
    assert _FakeEngine.instances[0].display_rows == 2
    ctrl.stop()
    ctrl.set_display_rows(5)   # 沒在跑也不能炸


def test_stop_is_idempotent_and_emits_idle_once(tmp_path):
    cfg, ctrl = _controller(tmp_path)
    states = []
    ctrl.state_changed.connect(lambda s, m: states.append(s))
    ctrl.start()
    ctrl.stop()
    ctrl.stop()
    assert states.count("idle") == 1


def test_real_engine_thread_starts_and_stops_quickly(tmp_path):
    """不用假引擎：確認 start/stop 真的起執行緒且 stop 不會卡 GUI。"""
    cfg = Config(tmp_path / "config.json")
    ctrl = SystemCaptionsController(cfg, _FakeStt(), lambda: False)
    ctrl._capture_factory = _FakeCapture
    ctrl.start()
    t0 = time.monotonic()
    ctrl.stop()
    assert time.monotonic() - t0 < 0.5
```

- [ ] **Step 3: 跑測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_system_captions.py tests/test_config.py -q`
Expected: FAIL（`rows_changed` 不存在、`display_rows` 為 None 等）

- [ ] **Step 4: 重寫 controller**

`app/core/system_captions.py` 整個換成：

```python
from PySide6.QtCore import QObject, Signal

from .local_translate import LocalTranslator
from .streaming_captions import DISPLAY_ROWS, StreamingCaptionEngine
from .system_audio import RollingAudioBuffer, SystemAudioCapture


class SystemCaptionsController(QObject):
    """組裝「系統聲音擷取 → 滾動緩衝 → 串流引擎 → 本機翻譯」，把回呼轉成 Qt 訊號。

    世代編號：每次 start() 遞增；stop() 之後舊 capture／engine 的回呼一律忽略。
    """

    rows_changed = Signal(object)          # list[Row]
    sentence_finalized = Signal(str, str)  # 完成句 (原文, 翻譯) → 歷史面板
    state_changed = Signal(str, str)       # state, message
    fatal_error = Signal(str)              # 擷取死掉、模型載不起來 → 關閉功能

    def __init__(self, config, stt, mic_busy, parent=None):
        super().__init__(parent)
        self.config = config
        self.stt = stt
        self._mic_busy = mic_busy
        self._translator = LocalTranslator(
            engine=config.get("system_captions", "engine", default="nllb-600m"),
            compute_device=config.get("system_captions", "compute_device",
                                      default="auto"))
        self._capture_factory = SystemAudioCapture
        self._engine_factory = StreamingCaptionEngine
        self._capture = None
        self._engine = None
        self._buffer = None
        self._running = False
        self._generation = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._running = True
        self._generation += 1
        gen = self._generation
        self._translator.set_engine(
            self.config.get("system_captions", "engine", default="nllb-600m"))
        self._translator.set_compute_device(
            self.config.get("system_captions", "compute_device", default="auto"))
        self._buffer = RollingAudioBuffer()
        buffer = self._buffer
        self._engine = self._engine_factory(
            buffer, self.stt, self._translator,
            languages=self._languages, mic_busy=self._mic_busy,
            on_rows=lambda rows, g=gen: self._guarded(g, self.rows_changed.emit, rows),
            on_state=lambda s, m, g=gen: self._guarded(g, self.state_changed.emit, s, m),
            on_fatal=lambda msg, g=gen: self._on_fatal(msg, g),
            on_final=lambda o, t, g=gen: self._guarded(g, self.sentence_finalized.emit, o, t),
            display_rows=self.config.get("system_captions", "display_rows",
                                         default=DISPLAY_ROWS))
        self._engine.start()
        self._capture = self._capture_factory(
            device_name=self.config.get("system_captions", "device",
                                        default="default"),
            on_frames=lambda frames, g=gen, b=buffer: self._on_frames(frames, g, b),
            on_error=lambda error, g=gen: self._on_fatal(
                f"系統聲音擷取失敗：{error}", g))
        self._capture.start()
        self.state_changed.emit("listening", "正在聽系統聲音…")

    def stop(self):
        if not self._running and self._capture is None and self._engine is None:
            return
        self._running = False
        self._generation += 1   # 先變號：舊回呼從此無效
        capture, self._capture = self._capture, None
        engine, self._engine = self._engine, None
        if capture is not None:
            capture.stop()      # join ≤ 0.2s
        if engine is not None:
            engine.stop()       # join ≤ 0.1s
        self.state_changed.emit("idle", "已停止系統聲音字幕")

    def set_display_rows(self, n: int):
        if self._engine is not None:
            self._engine.set_display_rows(n)

    # ---- 內部 ----

    def _languages(self):
        """(辨識語言, 翻譯目標語言)：預設辨識「目標語言」、翻成「母語」。"""
        native = self.config.get("language", "source", default="zh")
        target = self.config.get("language", "target", default="en")
        spoken = self.config.get("system_captions", "language", default="")
        return (spoken or target), native

    def _guarded(self, gen, emit, *args):
        if gen == self._generation and self._running:
            emit(*args)

    def _on_frames(self, frames, gen, buffer):
        if gen == self._generation and self._running:
            buffer.append(frames)

    def _on_fatal(self, message, gen):
        if gen != self._generation:
            return
        self._running = False
        self.fatal_error.emit(message)
        self.state_changed.emit("error", message)
```

- [ ] **Step 5: 跑測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_system_captions.py tests/test_config.py -q`
Expected: 全部 passed

- [ ] **Step 6: Commit**

```bash
git add app/core/system_captions.py app/config.py config.example.json tests/test_system_captions.py tests/test_config.py
git commit -m "refactor: SystemCaptionsController 改為組裝串流引擎；config 以 display_rows 取代切段參數"
```

---

### Task 6: SystemSubtitleOverlay.set_rows

**Files:**
- Modify: `app/ui/system_subtitle.py`
- Test: `tests/test_system_subtitle.py`（新）

**Interfaces:**
- Consumes: `Row` from `app.core.streaming_captions`
- Produces: `SystemSubtitleOverlay.set_rows(rows: list[Row])`、`add_history(original, translated)`；移除 `show_source`/`update_caption`；屬性 `row_widgets: list[tuple[QLabel, QLabel]]`（測試用；長度 = 目前顯示行數）

- [ ] **Step 1: 寫失敗測試**

建立 `tests/test_system_subtitle.py`：

```python
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config  # noqa: E402
from app.core.streaming_captions import Row  # noqa: E402
from app.ui.system_subtitle import SystemSubtitleOverlay  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _overlay(tmp_path, qapp):
    return SystemSubtitleOverlay(Config(tmp_path / "config.json"))


def test_set_rows_creates_one_widget_pair_per_row(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.set_rows([Row("One.", "一。", True), Row("Two", "", False)])
    assert len(ov.row_widgets) == 2
    assert ov.row_widgets[0][0].text() == "One."
    assert ov.row_widgets[0][1].text() == "一。"
    assert ov.row_widgets[1][0].text() == "Two"
    assert ov.row_widgets[1][1].text() == "…"      # 未完句尚無翻譯


def test_set_rows_shrinks_and_grows(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.set_rows([Row("a"), Row("b"), Row("c")])
    assert len(ov.row_widgets) == 3
    ov.set_rows([Row("z", "乙", True)])
    assert len(ov.row_widgets) == 1
    assert ov.row_widgets[0][0].text() == "z"


def test_history_accumulates_finals_only_via_add_history(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.add_history("Hello.", "你好。")
    ov.add_history("Bye.", "再見。")
    ov._refresh_history()
    text = ov.history_view.toPlainText()
    assert "Hello." in text and "再見。" in text
    ov.clear_history()
    ov._refresh_history()
    assert ov.history_view.toPlainText() == ""


def test_apply_style_survives_row_count_change(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.set_rows([Row("a"), Row("b")])
    ov.config.set("system_captions", "font_size", 30)
    ov.apply_style()
    assert ov.row_widgets[1][1].font().pixelSize() == 30
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_system_subtitle.py -q`
Expected: FAIL，`AttributeError: ... has no attribute 'set_rows'`

- [ ] **Step 3: 實作**

在 `app/ui/system_subtitle.py`：

(a) `__init__` 中，把

```python
        self.original_label = QLabel("", self)
        self.original_label.setWordWrap(True)
        outer.addWidget(self.original_label)

        self.translated_label = QLabel("", self)
        self.translated_label.setWordWrap(True)
        outer.addWidget(self.translated_label)
        outer.addStretch(1)
```

換成

```python
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setSpacing(4)
        outer.addLayout(self.rows_layout)
        self.row_widgets = []   # [(original_label, translated_label), ...]
        outer.addStretch(1)
```

(b) 刪掉 `show_source` 與 `update_caption` 兩個方法，換成：

```python
    def set_rows(self, rows):
        """顯示最近幾行：每行原文小字 + 翻譯大字；未完句沒翻譯時顯示「…」。"""
        while len(self.row_widgets) < len(rows):
            original = QLabel("", self)
            original.setWordWrap(True)
            translated = QLabel("", self)
            translated.setWordWrap(True)
            self.rows_layout.addWidget(original)
            self.rows_layout.addWidget(translated)
            self.row_widgets.append((original, translated))
        while len(self.row_widgets) > len(rows):
            original, translated = self.row_widgets.pop()
            self.rows_layout.removeWidget(original)
            self.rows_layout.removeWidget(translated)
            original.deleteLater()
            translated.deleteLater()
        for (original, translated), row in zip(self.row_widgets, rows):
            original.setText(row.original)
            translated.setText(row.translated or ("…" if not row.is_final else ""))
        self.apply_style()

    def add_history(self, original: str, translated: str):
        self._history.append((original, translated))
        if self.history_view.isVisible():
            self._refresh_history()
```

(c) `apply_style` 改成對每一行套用：

```python
    def apply_style(self):
        size = int(self.config.get(
            self.CONFIG_SECTION, "font_size", default=20))
        original_font = QFont("Segoe UI")
        original_font.setPixelSize(max(12, round(size * 0.8)))
        translated_font = QFont("Microsoft JhengHei")
        translated_font.setPixelSize(size)
        translated_font.setBold(True)
        last = len(self.row_widgets) - 1
        for i, (original, translated) in enumerate(self.row_widgets):
            original.setFont(original_font)
            translated.setFont(translated_font)
            # 舊句淡一點，正在講的那句最亮
            dim = i < last
            original.setStyleSheet("color: #8fb8c2;" if dim else "color: #cfe9ef;")
            translated.setStyleSheet("color: #d0d0d0;" if dim else "color: white;")
        self.update()
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_system_subtitle.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/ui/system_subtitle.py tests/test_system_subtitle.py
git commit -m "feat: 系統字幕疊加層改為多行顯示（set_rows），舊句淡化、未完句顯示…"
```

---

### Task 7: 接線——主視窗、設定頁、文件

**Files:**
- Modify: `app/ui/main_window.py`（`_on_system_source`/`_on_system_caption` → `_on_system_rows`/`_on_system_final`；display_rows 即時生效）
- Modify: `app/ui/settings_page.py`（刪兩個 spin，加「顯示行數」）
- Modify: `README.md`
- Test: 既有測試全綠 + `python -c "import app.ui.main_window"` 成功

**Interfaces:**
- Consumes: `SystemCaptionsController.rows_changed/sentence_finalized/set_display_rows`（Task 5）、`SystemSubtitleOverlay.set_rows/add_history`（Task 6）

- [ ] **Step 1: main_window.py**

把

```python
        self.system_captions.caption_ready.connect(self._on_system_caption)
        self.system_captions.source_ready.connect(self._on_system_source)
```

換成

```python
        self.system_captions.rows_changed.connect(self._on_system_rows)
        self.system_captions.sentence_finalized.connect(self._on_system_final)
```

把

```python
    def _on_system_source(self, original: str):
        self.system_subtitle.show_source(original)
        self._avoid_overlap(self.system_subtitle, self.subtitle)

    def _on_system_caption(self, original: str, translated: str):
        self.system_subtitle.update_caption(original, translated)
        self._avoid_overlap(self.system_subtitle, self.subtitle)
```

換成

```python
    def _on_system_rows(self, rows):
        self.system_subtitle.set_rows(rows)
        self._avoid_overlap(self.system_subtitle, self.subtitle)

    def _on_system_final(self, original: str, translated: str):
        self.system_subtitle.add_history(original, translated)
```

`_apply_system_caption_style` 改成：

```python
    def _apply_system_caption_style(self):
        self.system_subtitle.apply_style()
        self.system_subtitle.apply_opacity()
        self.system_captions.set_display_rows(
            self.config.get("system_captions", "display_rows", default=3))
```

- [ ] **Step 2: settings_page.py**

(a) 建立控件：把 `self.system_segment_spin = ...` 到 `self._add_row(layout, "停頓多久算一句", self.system_pause_spin)` 那 10 行換成

```python
        self.system_rows_spin = SpinBox(self.view)
        self.system_rows_spin.setRange(1, 5)
        self.system_rows_spin.setSuffix(" 行")
        self._add_row(layout, "同時顯示幾行（最近幾句 + 正在講的這句）",
                      self.system_rows_spin)
```

(b) 載入：把

```python
        self.system_segment_spin.setValue(cfg.get(sc, "max_segment_sec", default=4))
        self.system_pause_spin.setValue(
            cfg.get(sc, "segment_silence_ms", default=400))
```

換成

```python
        self.system_rows_spin.setValue(cfg.get(sc, "display_rows", default=3))
```

(c) 訊號連接：把 `system_segment_spin.valueChanged` 與 `system_pause_spin.valueChanged` 兩段 connect 換成

```python
        self.system_rows_spin.valueChanged.connect(
            self._on_system_settings_changed)
```

(d) `_on_system_settings_changed`：`pipeline_keys` 改回 `("device", "engine", "compute_device")`；把

```python
        self.config.set(sc, "max_segment_sec", self.system_segment_spin.value())
        self.config.set(sc, "segment_silence_ms", self.system_pause_spin.value())
```

換成

```python
        self.config.set(sc, "display_rows", self.system_rows_spin.value())
```

- [ ] **Step 3: README**

`README.md` 系統聲音字幕小節，把

```
- 為了即時性，每段最長 4 秒、停頓 0.4 秒就切句（設定頁可調）；原文先出現，翻譯完成再補上。
  想更快就把「每段最長」調短，想句子完整就調長。
```

換成

```
- Live Captions 風格：每秒重新辨識「上一句句尾之後」的聲音，正在講的句子邊講邊長；
  文字穩定約 2 秒或講完一句就翻譯，文字變了會重翻，相同不重翻。
- 疊加層顯示最近 3 行（設定頁 1–5 行），新句進來舊句往上頂掉；正在講的那句最亮。
```

架構重點那段的 `能量門檻切句後才送辨識` 改成 `連續音框進滾動緩衝，串流引擎（app/core/streaming_captions.py）每秒重新辨識開放窗`。

- [ ] **Step 4: 驗證**

Run: `.venv\Scripts\python.exe -m pytest tests -q`
Expected: 全部 passed（0 failed）

Run: `.venv\Scripts\python.exe -c "import app.ui.main_window, app.ui.settings_page; print('ok')"`
Expected: `ok`

Run: `grep -rn "segment_silence_ms\|max_segment_sec\|caption_ready\|source_ready\|show_source\|update_caption\|SegmentAccumulator" app/ tests/ README.md config.example.json`
Expected: 無輸出

- [ ] **Step 5: Commit**

```bash
git add app/ui/main_window.py app/ui/settings_page.py README.md
git commit -m "feat: 主視窗／設定頁接上串流字幕（多行顯示、顯示行數即時生效）"
```
