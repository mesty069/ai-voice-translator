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
