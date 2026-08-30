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
