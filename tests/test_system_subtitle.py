import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config  # noqa: E402
from app.core.streaming_captions import VERYLONG_THRESHOLD, Row  # noqa: E402
from app.ui.system_subtitle import (  # noqa: E402
    ACTIVE_TEXT,
    DIM_TEXT,
    SystemSubtitleOverlay,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _overlay(tmp_path, qapp):
    return SystemSubtitleOverlay(Config(tmp_path / "config.json"))


def test_set_rows_shows_one_translation_paragraph_and_current_original(tmp_path, qapp):
    """照 OverlayWindow.xaml：上方是接成一段的翻譯，下方只有當前句原文。"""
    ov = _overlay(tmp_path, qapp)
    ov.set_rows([Row("One.", "一。", True), Row("Two.", "二。", True),
                 Row("Thr", "", False)])
    text = ov.translation_label.text()
    assert "一。" in text and "二。" in text
    assert text.index("一。") < text.index("二。")
    assert "…" not in text                  # 當前句還沒翻好就不顯示佔位符
    assert ov.original_label.text() == "Thr"


def test_previous_translations_are_dimmer_than_the_current_one(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.set_rows([Row("One.", "一。", True), Row("Two", "二", False)])
    text = ov.translation_label.text()
    assert DIM_TEXT in text and ACTIVE_TEXT in text
    assert text.index(DIM_TEXT) < text.index(ACTIVE_TEXT)


def test_translation_paragraph_joins_cjk_without_space(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.set_rows([Row("One.", "你好。", True), Row("Two", "世界", False)])
    assert "你好。</span><span" in ov.translation_label.text()


def test_set_rows_escapes_html_in_text(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.set_rows([Row("a < b", "甲 < 乙", True), Row("c > d", "丙 > 丁", False)])
    text = ov.translation_label.text()
    assert "甲 &lt; 乙" in text and "丙 &gt; 丁" in text


def test_set_rows_skips_previous_rows_without_translation(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.set_rows([Row("One.", "", True), Row("Two.", "二。", True),
                 Row("Thr", "三", False)])
    text = ov.translation_label.text()
    assert "二。" in text and "三" in text


def test_set_rows_with_no_rows_clears_both_labels(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.set_rows([Row("One.", "一。", True), Row("Two", "二", False)])
    ov.set_rows([])
    assert ov.translation_label.text() == ""
    assert ov.original_label.text() == ""


def test_last_row_is_the_current_sentence_even_when_final(tmp_path, qapp):
    """句尾之後那句仍留在下方原文，直到新的字出現（它就是這樣）。"""
    ov = _overlay(tmp_path, qapp)
    ov.set_rows([Row("One.", "一。", True), Row("Two.", "二。", True)])
    assert ov.original_label.text() == "Two."
    assert ACTIVE_TEXT in ov.translation_label.text()


def test_very_long_original_is_shortened(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    tail = "b" * 230
    ov.set_rows([Row("A, " + tail, "甲", False)])
    assert ov.original_label.text() == tail
    assert len(ov.original_label.text().encode("utf-8")) >= VERYLONG_THRESHOLD


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


def test_apply_style_uses_font_size_for_translation_and_smaller_original(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.set_rows([Row("a", "甲", True), Row("b", "乙", False)])
    ov.config.set("system_captions", "font_size", 30)
    ov.apply_style()
    assert ov.translation_label.font().pixelSize() == 30
    assert ov.original_label.font().pixelSize() == round(30 * 0.8)


def test_preferred_height_grows_with_rows_and_font(tmp_path, qapp):
    """C1：高度要跟著行數與字級長。"""
    ov = _overlay(tmp_path, qapp)
    assert ov.preferred_height(1) >= 130
    assert ov.preferred_height(3) > ov.preferred_height(1)
    small = ov.preferred_height(3)
    ov.config.set("system_captions", "font_size", 30)
    assert ov.preferred_height(3) > small


def test_set_rows_grows_overlay_to_fit_three_rows(tmp_path, qapp):
    """C1：三行字幕不能被裁掉，要自己長高。"""
    ov = _overlay(tmp_path, qapp)
    ov.resize(360, 130)
    ov.set_rows([Row("One.", "一。", True), Row("Two.", "二。", True),
                 Row("Three", "", False)])
    assert ov.height() >= ov.preferred_height(3)
    assert ov.width() == 360      # 只長高、不動寬


def test_set_rows_never_shrinks_user_resized_overlay(tmp_path, qapp):
    """C1：使用者自己拉大的高度不能被縮回去。"""
    ov = _overlay(tmp_path, qapp)
    tall = ov.preferred_height(3) + 200
    ov.resize(400, tall)
    ov.set_rows([Row("One.", "一。", True)])
    assert ov.height() == tall


def test_show_overlay_without_saved_size_fits_display_rows(tmp_path, qapp):
    """C1：第一次顯示就要放得下設定的行數。"""
    ov = _overlay(tmp_path, qapp)
    ov.config.set("system_captions", "display_rows", 3)
    ov.show_overlay()
    assert ov.height() >= ov.preferred_height(3)
    ov.hide()


def test_wrapped_long_text_is_not_clipped(tmp_path, qapp):
    """長句換行後，視窗高度必須放得下所有 label（使用者回報被裁掉）。"""
    ov = _overlay(tmp_path, qapp)
    ov.show()   # 隱藏的 widget 不會收到 resizeEvent（Qt 延後到 show）
    ov.resize(600, ov.preferred_height(3))
    long_en = "This is a deliberately long English sentence that will certainly wrap " * 3
    long_zh = "這是一句故意寫得非常長、一定會換行的中文翻譯結果，用來確認字幕框會跟著長高。" * 2
    ov.set_rows([Row(long_en, long_zh, True), Row(long_en, long_zh, True),
                 Row("short", "短", False)])
    qapp.processEvents()
    assert ov.height() >= ov.layout().heightForWidth(ov.width())
    assert ov.height() > ov.preferred_height(3)      # 換行確實比估值高
    # 拉窄 → 折更多行 → 再長高
    before = ov.height()
    ov.resize(360, before)   # MIN_SIZE 寬 360，不能再窄
    qapp.processEvents()
    assert ov.width() == 360
    assert ov.height() >= ov.layout().heightForWidth(360)
    assert ov.height() > before


def test_history_replaces_entry_when_sentence_is_merged(tmp_path, qapp):
    """短句併進前一句後再送一次合併句 → 逐字稿只留合併後那一筆。"""
    ov = _overlay(tmp_path, qapp)
    ov.add_history("Please join now.", "請現在加入我們。")
    ov.add_history("Please join now. Now.", "請現在加入我們，現在。")
    ov.add_history("Thank you.", "謝謝你。")
    assert ov._history == [("Please join now. Now.", "請現在加入我們，現在。"),
                           ("Thank you.", "謝謝你。")]
