import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config  # noqa: E402
from app.ui import subtitle as subtitle_mod  # noqa: E402
from app.ui.subtitle import SubtitleOverlay  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _overlay(tmp_path, qapp):
    return SubtitleOverlay(Config(tmp_path / "config.json"))


def _chip_texts(ov):
    return [chip.text() for chip in ov.word_chips()]


# ---- 單字方塊 ----

def test_show_result_makes_one_chip_per_word(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.show_result("你好 世界", "Hello there world")
    assert _chip_texts(ov) == ["Hello", "there", "world"]


def test_punctuation_sticks_to_the_previous_word(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.show_result("你好，世界。", "Hello, world.")
    assert _chip_texts(ov) == ["Hello,", "world."]


def test_clicking_chip_speaks_from_that_word_onward(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    spoken = []
    ov.speak_requested.connect(spoken.append)
    ov.show_result("你好", "Hello there big world.")
    ov.word_chips()[2].click()
    assert spoken == ["big world."]
    ov.word_chips()[0].click()
    assert spoken[-1] == "Hello there big world."


def test_clicking_chip_restarts_the_countdown(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.show_result("你好", "Hello world")
    ov._hide_timer.stop()
    ov.word_chips()[0].click()
    assert ov._hide_timer.isActive()


def test_show_message_clears_the_chips(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.show_result("你好", "Hello world")
    assert ov.word_chips()
    ov.show_message("⚠ 沒有偵測到語音")
    assert ov.word_chips() == []
    assert not ov.words_widget.isVisible()


def test_result_without_english_has_no_chips(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.show_result("你好", "")
    assert ov.word_chips() == []


def test_chips_use_the_english_font_size(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.config.set("subtitle", "font_size", 30)
    ov.show_result("你好", "Hello world")
    assert ov.word_chips()[0].font().pixelSize() == 30


# ---- 暫停大按鈕 ----

def test_pause_button_toggles_pause_and_resume(tmp_path, qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(subtitle_mod.tts, "pause",
                        lambda: calls.append(("pause",)) or True)
    monkeypatch.setattr(subtitle_mod.tts, "resume",
                        lambda volume=None: calls.append(("resume", volume))
                        or True)
    ov = _overlay(tmp_path, qapp)
    ov.show_result("你好", "Hello world")
    ov.set_reading(True)
    assert ov.pause_button.isEnabled()
    ov.pause_button.click()
    assert calls == [("pause",)]
    assert ov.is_paused is True
    ov.pause_button.click()
    assert calls[1] == ("resume", 1.0)   # config 預設音量 100% → 1.0
    assert ov.is_paused is False


def test_pause_button_disabled_unless_reading(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.show_result("你好", "Hello world")
    assert not ov.pause_button.isEnabled()
    ov.set_reading(True)
    assert ov.pause_button.isEnabled()
    ov.set_reading(False)
    assert not ov.pause_button.isEnabled()


def test_new_reading_clears_the_paused_state(tmp_path, qapp, monkeypatch):
    monkeypatch.setattr(subtitle_mod.tts, "pause", lambda: True)
    ov = _overlay(tmp_path, qapp)
    ov.show_result("你好", "Hello world")
    ov.set_reading(True)
    ov.pause_button.click()
    assert ov.is_paused is True
    ov.set_reading(True)      # 新的一句開始朗讀
    assert ov.is_paused is False


def test_pause_button_hidden_without_english(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.show_message("⚠ 出錯了")
    assert ov.pause_button.isHidden()


# ---- 音量滑桿 ----

def test_volume_slider_syncs_from_config(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.config.set("output", "tts_volume", 55)
    ov.show_result("你好", "Hello world")
    assert ov.volume_slider.value() == 55
    assert "55" in ov.volume_label.text()


def test_volume_slider_writes_config_and_emits(tmp_path, qapp):
    ov = _overlay(tmp_path, qapp)
    ov.show_result("你好", "Hello world")
    seen = []
    ov.volume_changed.connect(seen.append)
    ov.volume_slider.setValue(40)
    assert ov.config.get("output", "tts_volume") == 40
    assert seen == [40]
    assert "音量" in ov.volume_label.text()
    assert "40" in ov.volume_label.text()
