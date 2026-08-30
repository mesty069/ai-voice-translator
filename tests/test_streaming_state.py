import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.stt import Word  # noqa: E402
from app.core.streaming_captions import (  # noqa: E402
    IDLE_SEC,
    MAX_SYNC,
    SHORT_THRESHOLD,
    VERYLONG_THRESHOLD,
    CaptionState,
    Row,
    Sentence,
    join_words,
    shorten_display_sentence,
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

def test_current_grows_and_translates_after_idle_seconds():
    """照 LiveCaptions-Translator 的 MaxIdleInterval：文字停住 IDLE_SEC 就翻。"""
    st = CaptionState(display_rows=3)
    assert st.update_current("The meeting", 0.0) is False
    assert st.rows == [Row("The meeting", "", False)]
    assert st.update_current("The meeting", 1.0) is False       # 還沒停夠久
    assert st.update_current("The meeting", IDLE_SEC + 0.05) is True
    st.set_current_translation("會議")
    assert st.rows == [Row("The meeting", "會議", False)]


def test_idle_timer_restarts_when_text_changes():
    st = CaptionState()
    assert st.update_current("The meeting", 0.0) is False
    assert st.update_current("The meeting will", 1.2) is False   # 變了 → 重新計時
    assert st.update_current("The meeting will", 2.0) is False
    assert st.update_current("The meeting will", 2.5) is True


def test_same_text_is_not_retranslated():
    st = CaptionState()
    assert st.update_current("Stable text", 0.0) is False
    assert st.update_current("Stable text", 2.0) is True
    st.set_current_translation("穩定")
    for i in range(5):
        assert st.update_current("Stable text", 3.0 + i) is False


def test_sync_count_translates_after_max_sync_changes():
    """照 MaxSyncInterval：文字連續變 MAX_SYNC+1 次（每次都不算短句）就翻。"""
    st = CaptionState()
    texts = ["The meeting", "The meeting will", "The meeting will begin",
             "The meeting will begin now"]
    assert len(texts) == MAX_SYNC + 1
    for i, text in enumerate(texts[:-1]):
        assert st.update_current(text, i * 0.1) is False
    assert st.update_current(texts[-1], (len(texts) - 1) * 0.1) is True
    # 觸發後歸零：接下來又要再變 MAX_SYNC+1 次
    st.set_current_translation("會議即將開始")
    for i, text in enumerate(["a" * 12, "a" * 13, "a" * 14]):
        assert st.update_current(text, 1.0 + i * 0.1) is False
    assert st.update_current("a" * 15, 1.3) is True


def test_short_text_changes_never_trigger_sync():
    """短文字（UTF-8 位元組 < SHORT_THRESHOLD）變再多次也不因 sync 觸發。"""
    st = CaptionState()
    for i in range(MAX_SYNC + 4):
        text = "a" * (i + 1)                 # 最長 7 位元組，仍是短句
        assert len(text.encode("utf-8")) < SHORT_THRESHOLD
        assert st.update_current(text, i * 0.1) is False


def test_eos_translates_immediately():
    st = CaptionState()
    assert st.update_current("Done.", 0.0) is True


def test_translation_kept_while_text_grows():
    st = CaptionState()
    st.update_current("Hello", 0.0)
    st.set_current_translation("你好")
    st.update_current("Hello there", 0.1)
    assert st.rows[-1] == Row("Hello there", "你好", False)


def test_commit_text_pushes_rows_and_caps_display():
    """display_rows 是「保留前幾句」，rows 另外再帶上當前句 → display_rows + 1 列。"""
    # 句子都刻意寫長（>= SHORT_THRESHOLD 位元組），免得被短句合併規則併成一列
    st = CaptionState(display_rows=3)
    st.update_current("partial", 0.0)
    r1 = st.commit_text("Sentence one.")
    r1.translated = "第一句。"
    assert st.rows == [Row("Sentence one.", "第一句。", True)]     # current 清掉了
    st.commit_text("Sentence two.")
    st.commit_text("Sentence three.")
    st.commit_text("Sentence four.")
    st.update_current("Five", 0.0)
    assert [r.original for r in st.rows] == [
        "Sentence two.", "Sentence three.", "Sentence four.", "Five"]
    st.set_display_rows(2)
    assert [r.original for r in st.rows] == [
        "Sentence three.", "Sentence four.", "Five"]


def test_rows_keeps_display_rows_previous_sentences_plus_current():
    st = CaptionState(display_rows=1)
    st.commit_text("Sentence one.")
    st.commit_text("Sentence two.")
    st.update_current("Three", 0.0)
    assert [r.original for r in st.rows] == ["Sentence two.", "Three"]


# ---- shorten_display_sentence ----

def test_shorten_display_sentence_keeps_short_text():
    assert shorten_display_sentence("Hello there.", VERYLONG_THRESHOLD) == \
        "Hello there."


def test_shorten_display_sentence_cuts_before_first_punctuation():
    tail = "b" * 230
    assert shorten_display_sentence("A, " + tail, VERYLONG_THRESHOLD) == tail


def test_shorten_display_sentence_without_punctuation_is_unchanged():
    text = "c" * 300
    assert shorten_display_sentence(text, VERYLONG_THRESHOLD) == text


def test_commit_current_keeps_translation():
    st = CaptionState()
    st.update_current("Trailing", 0.0)
    st.set_current_translation("尾句")
    row = st.commit_current()
    assert row == Row("Trailing", "尾句", True)
    assert st.rows == [row]
    assert st.commit_current() is None


def test_is_short_counts_utf8_bytes():
    """照 LiveCaptions-Translator：門檻是 UTF-8 位元組數，中日韓約 3 個字。"""
    assert CaptionState.is_short("Yes.") is True
    assert CaptionState.is_short("x" * SHORT_THRESHOLD) is False
    assert len("你好".encode("utf-8")) < SHORT_THRESHOLD
    assert CaptionState.is_short("你好") is True
    assert len("你好嗎？".encode("utf-8")) >= SHORT_THRESHOLD
    assert CaptionState.is_short("你好嗎？") is False
    assert CaptionState.is_short("") is True


def test_commit_target_merges_short_text_with_previous_final():
    st = CaptionState()
    assert st.commit_target("Yes.") == "Yes."          # 還沒有前一句
    st.commit_text("Previous sentence here.")
    assert st.commit_target("Yes.") == "Previous sentence here. Yes."
    assert st.commit_target("x" * SHORT_THRESHOLD) == "x" * SHORT_THRESHOLD


def test_commit_target_uses_join_words_for_cjk():
    st = CaptionState()
    st.commit_text("這是前面一句話。")
    assert st.commit_target("好。") == "這是前面一句話。好。"


def test_commit_text_merges_short_sentence_into_previous_row():
    """短句要併進前一列，不能自己另起一列（否則前一句的翻譯會出現兩次）。"""
    st = CaptionState()
    first = st.commit_text("Previous sentence here.")
    first.translated = "前面那句。"
    merged = st.commit_text("Yes.")
    assert merged is first                     # 同一個 Row 物件被就地改寫
    assert len(st.rows) == 1
    assert st.rows[0] == Row("Previous sentence here. Yes.", "", True)


def test_commit_text_does_not_merge_long_sentence():
    st = CaptionState()
    st.commit_text("Previous sentence here.")
    st.commit_text("Another long sentence.")
    assert [r.original for r in st.rows] == [
        "Previous sentence here.", "Another long sentence."]


def test_commit_current_merges_short_pending_into_previous_row():
    st = CaptionState()
    st.commit_text("Previous sentence here.")
    st.update_current("Yes", 0.0)
    st.set_current_translation("是")
    row = st.commit_current()
    assert row.original == "Previous sentence here. Yes"
    assert row.is_final is True
    assert st.rows == [Row("Previous sentence here. Yes", "", True)]
    assert st.has_current is False


def test_rows_returns_copies():
    st = CaptionState()
    st.update_current("a", 0.0)
    st.rows[0].original = "mutated"
    assert st.rows[0].original == "a"


def test_has_current_tracks_pending_row():
    """A2：辨識器這輪沒回字時，引擎仍要知道有未完句待收。"""
    st = CaptionState()
    assert st.has_current is False
    st.update_current("partial", 0.0)
    assert st.has_current is True
    st.commit_current()
    assert st.has_current is False


def test_current_row_is_a_copy_of_the_pending_row():
    """F1：引擎要在 commit 之前先讀目前句（翻譯失敗時原句才留得住）。"""
    st = CaptionState()
    assert st.current_row is None
    st.update_current("Trailing", 0.0)
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
