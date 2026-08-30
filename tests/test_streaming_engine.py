import sys
import threading
import time
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


def _engine(stt, translator=None, buffer=None, sink=None, mic_busy=lambda: False,
            now=None, sleep=None):
    sink = sink or _Sink()
    buffer = buffer or RollingAudioBuffer()
    eng = StreamingCaptionEngine(
        buffer, stt, translator or _FakeTranslator(),
        languages=lambda: ("en", "zh"), mic_busy=mic_busy,
        on_rows=lambda rows: sink.rows.append(rows),
        on_state=lambda s, m: sink.states.append((s, m)),
        on_fatal=sink.fatals.append,
        on_final=lambda o, t: sink.finals.append((o, t)),
        now=now or (lambda: 0.0), sleep=sleep or (lambda s: None))
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
    # 推到「下一個還沒收的字」的起點 1.2，不是句尾字的 end 0.8：
    # end 常被 DTW 抓得偏早，停在那裡的話殘音下一輪會再被
    # 辨識成碎片句（例如 "there."），歷史就多一列重複。
    assert abs(eng.committed_t - 1.2) < 1e-6
    assert tr.calls == ["Hello there."]
    assert sink.finals == [("Hello there.", "譯[Hello there.]")]
    rows = sink.rows[-1]
    assert rows[0] == rows[0].__class__("Hello there.", "譯[Hello there.]", True)
    assert rows[1].original == "Next" and rows[1].is_final is False
    # 下一輪只送 committed_t 之後的音訊
    eng.step()
    assert stt.calls[-1][0] == int(SAMPLE_RATE * 2.0) - round(SAMPLE_RATE * 1.2)
    assert stt.calls[-1][2] == 1                 # beam_size=1：串流要快
    assert abs(buf.start_seconds - 1.2) < 1e-6   # 已 trim


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
    assert abs(eng.committed_t - 1.2) < 1e-6


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


# ---- 修復波 1 ----

class _StoppingTranslator:
    """在 translate() 裡把引擎 stop() 掉，模擬「翻譯很慢、使用者中途關閉」。"""

    def __init__(self, raise_after=None):
        self.engine = None
        self.calls = []
        self.raise_after = raise_after

    def translate(self, text, src, tgt, progress_cb=None):
        self.calls.append(text)
        self.engine.stop()
        if self.raise_after is not None:
            raise self.raise_after
        return f"譯[{text}]"


def test_short_current_committed_by_silence_uses_previous_final():
    """A1：靜音收句的短尾句要接「前一句」，不能接到自己。"""
    words1 = [Word("A", 0.0, 0.1), Word("fairly", 0.2, 0.4),
              Word("long", 0.5, 0.7), Word("sentence.", 0.8, 1.0)]
    words2 = [Word("Yes", 0.0, 0.2)]     # 沒句尾標點 → 未完句
    stt = _ScriptedStt([words1, words2])
    tr = _FakeTranslator()
    eng, buf, sink = _engine(stt, tr)
    buf.append(_speech(1.5))
    eng.step()
    buf.append(_silence(SILENCE_COMMIT_SEC + 0.1))
    eng.step()
    assert tr.calls[-1] == "A fairly long sentence. Yes"
    assert sink.finals[-1][0] == "Yes"


def test_pending_current_finalizes_when_stt_returns_empty():
    """A2：辨識器對尾端靜音回空 words 時，未完句仍要收成完成句。"""
    stt = _ScriptedStt([[Word("Hello", 0.0, 0.4)], []])
    eng, buf, sink = _engine(stt)
    buf.append(_speech(1.0))
    eng.step()
    assert sink.finals == []
    buf.append(_silence(SILENCE_COMMIT_SEC + 0.1))
    eng.step()
    assert sink.finals == [("Hello", "譯[Hello]")]
    assert sink.rows[-1][-1].is_final is True
    assert abs(eng.committed_t - buf.total_seconds) < 1e-6


def test_pending_current_finalized_before_dropping_stale_audio():
    """A2：窗太長又辨識不出字而丟音訊之前，先把未完句收掉。"""
    stt = _ScriptedStt([[Word("Talk", 0.0, 0.4)], []])
    eng, buf, sink = _engine(stt)
    buf.append(_speech(1.0))
    eng.step()
    buf.append(_speech(WINDOW_MAX_SEC + 0.5))   # 有音量 → 不是靜音路徑
    eng.step()
    assert sink.finals == [("Talk", "譯[Talk]")]
    assert abs(eng.committed_t - buf.total_seconds) < 1e-6


def test_stop_inside_translate_suppresses_final():
    """A3：stop() 之後不得再冒出完成句回呼。"""
    stt = _ScriptedStt([[Word("Done.", 0.0, 0.3)]])
    tr = _StoppingTranslator()
    eng, buf, sink = _engine(stt, tr)
    tr.engine = eng
    buf.append(_speech(1.0))
    eng.step()
    assert tr.calls == ["Done."]
    assert sink.finals == []
    assert sink.rows == []


def test_stop_inside_translate_suppresses_fatal():
    """A3：stop() 之後不得再冒出致命錯誤提示。"""
    stt = _ScriptedStt([[Word("Done.", 0.0, 0.3)]])
    tr = _StoppingTranslator(raise_after=ModelLoadError("no model"))
    eng, buf, sink = _engine(stt, tr)
    tr.engine = eng
    buf.append(_speech(1.0))
    eng.step()
    assert sink.fatals == []
    assert eng.is_running is False


def test_stop_from_engine_thread_does_not_join_itself():
    """致命錯誤是在引擎執行緒上回呼的，控制器會反手 stop() 引擎。"""
    eng, buf, sink = _engine(_ScriptedStt([[]]))
    eng._thread = threading.current_thread()
    eng.stop()                       # 不得丟 RuntimeError
    assert eng.is_running is False


def test_window_overflow_commit_skips_stale_audio():
    """A4：強制收句後窗仍過長的話，committed_t 要推到窗上限內，免得整句重來。"""
    stt = _ScriptedStt([[Word("Endless", 0.0, 0.5), Word("talk", 0.6, 1.0)]])
    eng, buf, sink = _engine(stt)
    buf.append(_speech(WINDOW_MAX_SEC + 8.0))
    eng.step()
    assert abs(eng.committed_t - (buf.total_seconds - WINDOW_MAX_SEC)) < 1e-6


def test_stt_wait_timeout_is_fatal():
    """A5：語音模型久久載不起來 → 致命錯誤、引擎停掉。"""
    stt = _ScriptedStt([[Word("x", 0.0, 0.1)]], ready=False)
    clock = [0.0]

    def now():
        clock[0] += 5.0
        return clock[0]

    eng, buf, sink = _engine(stt, now=now)
    eng.STT_WAIT_TIMEOUT = 4.0
    buf.append(_speech(1.0))
    assert eng.step() is False
    assert sink.fatals == []
    assert eng.step() is False
    assert sink.fatals and "逾時" in sink.fatals[0]
    assert eng.is_running is False


def test_start_twice_does_not_spawn_second_thread():
    """A6：連按兩次開始不能多起一條執行緒。"""
    eng, buf, sink = _engine(_ScriptedStt([[]]),
                             now=time.monotonic, sleep=time.sleep)
    eng.start()
    first = eng._thread
    eng.start()
    assert eng._thread is first
    eng.stop()
    first.join(1.0)
    assert first.is_alive() is False


# ---- 修復波 2 ----

class _FlakyTranslator:
    """指定第幾次呼叫要丟例外（模擬短暫的 GPU／記憶體錯誤）。"""

    def __init__(self, fail_on=(1,), error=None):
        self.calls = []
        self.attempts = 0
        self.fail_on = set(fail_on)
        self.error = error or RuntimeError("cuda oom")

    def translate(self, text, src, tgt, progress_cb=None):
        self.attempts += 1
        if self.attempts in self.fail_on:
            raise self.error
        self.calls.append(text)
        return f"譯[{text}]"


def test_completed_sentence_is_retried_when_translation_fails():
    """F1：翻譯失敗不得留下空翻譯的完成句，也不得推進 committed_t。"""
    stt = _ScriptedStt([[Word("Hello", 0.0, 0.3), Word("there.", 0.4, 0.8)]])
    tr = _FlakyTranslator()
    eng, buf, sink = _engine(stt, tr)
    buf.append(_speech(1.0))
    eng.step()
    assert sink.finals == []
    assert eng.state.rows == []                  # 沒有半成品的完成句
    assert abs(eng.committed_t) < 1e-6           # 音訊還在，下一輪重來
    assert sink.states[-1][0] == "error"
    assert sink.rows == []
    eng.step()                                   # 這次翻譯正常
    assert sink.finals == [("Hello there.", "譯[Hello there.]")]
    assert [r.original for r in eng.state.rows] == ["Hello there."]
    assert abs(eng.committed_t - 0.8) < 1e-6


def test_pending_sentence_is_retried_when_translation_fails():
    """F1：靜音收句時翻譯失敗 → 未完句留著，下一輪再收一次（不重複）。"""
    stt = _ScriptedStt([[Word("Trailing", 0.0, 0.4)], []])
    tr = _FlakyTranslator()
    eng, buf, sink = _engine(stt, tr)
    buf.append(_speech(1.0))
    eng.step()
    assert tr.calls == []                        # 還沒到翻譯門檻
    buf.append(_silence(SILENCE_COMMIT_SEC + 0.1))
    eng.step()
    assert sink.finals == []
    assert eng.state.rows[-1].original == "Trailing"
    assert eng.state.rows[-1].is_final is False  # 還是目前句
    assert abs(eng.committed_t) < 1e-6
    assert sink.states[-1][0] == "error"
    eng.step()                                   # 這次翻譯正常
    assert sink.finals == [("Trailing", "譯[Trailing]")]
    assert eng.state.rows[-1].is_final is True
    assert abs(eng.committed_t - buf.total_seconds) < 1e-6


def test_committed_t_stops_at_next_sentence_start_when_translation_fails():
    """F2：前一句收完、第二句翻譯失敗 → committed_t 停在第二句的起點。"""
    stt = _ScriptedStt([[Word("One.", 0.0, 0.4), Word("Two.", 1.0, 1.4)]])
    tr = _FlakyTranslator(fail_on=(2,))
    eng, buf, sink = _engine(stt, tr)
    buf.append(_speech(2.0))
    eng.step()
    assert sink.finals == [("One.", "譯[One.]")]
    assert abs(eng.committed_t - 1.0) < 1e-6     # 不是 "One." 的 end 0.4
