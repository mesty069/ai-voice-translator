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
    assert current == Sentence("Next one", 1.4, 0.9)


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
    assert current == Sentence("再", 0.9, 0.7)


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


def test_translate_input_accepts_explicit_previous():
    """A1：引擎在 commit 之前先取「前一句」，commit 之後才用它來翻譯。"""
    st = CaptionState()
    st.commit_text("Previous sentence here.")
    assert st.translate_input("Yes.", prev="Earlier one.") == "Earlier one. Yes."
    assert st.translate_input("Yes.", prev="") == "Yes."


def test_has_current_tracks_pending_row():
    """A2：辨識器這輪沒回字時，引擎仍要知道有未完句待收。"""
    st = CaptionState()
    assert st.has_current is False
    st.update_current("partial")
    assert st.has_current is True
    st.commit_current()
    assert st.has_current is False


def test_current_row_is_a_copy_of_the_pending_row():
    """F1：引擎要在 commit 之前先讀目前句（翻譯失敗時原句才留得住）。"""
    st = CaptionState()
    assert st.current_row is None
    st.update_current("Trailing")
    st.set_current_translation("尾句")
    row = st.current_row
    assert row == Row("Trailing", "尾句", False)
    row.original = "mutated"
    assert st.current_row.original == "Trailing"


def test_split_fills_sentence_start_from_first_word():
    """F2：引擎要把 committed_t 推到下一句的起點，所以 Sentence 得帶著 start。"""
    words = [_w("The", 0.5, 0.7), _w("end.", 0.8, 1.0),
             _w("Next", 1.6, 1.8), _w("one.", 1.9, 2.1),
             _w("More", 2.6, 2.8)]
    completed, current = split_sentences_by_words(words)
    assert [(s.text, s.start, s.end) for s in completed] == [
        ("The end.", 0.5, 1.0), ("Next one.", 1.6, 2.1)]
    assert (current.start, current.end) == (2.6, 2.8)


def test_sentence_start_defaults_to_zero():
    """start 是補在第三個欄位的，舊的 Sentence(text, end) 寫法要照舊可用。"""
    assert Sentence("x", 0.6) == Sentence("x", 0.6, 0.0)
