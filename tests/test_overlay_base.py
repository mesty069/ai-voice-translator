import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from app.config import Config
from app.ui.overlay_base import DraggableResizableOverlay


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Probe(DraggableResizableOverlay):
    CONFIG_SECTION = "subtitle"

    def __init__(self, config):
        super().__init__(config, Qt.WindowType.FramelessWindowHint)
        self.clicks = 0
        self.geometry_changes = 0
        self.resize(400, 200)

    def _on_simple_click(self, pos):
        self.clicks += 1

    def _on_geometry_changed(self):
        self.geometry_changes += 1


def _press(w, local, button=Qt.MouseButton.LeftButton):
    g = w.mapToGlobal(local)
    w.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(local.x(), local.y()),
        QPointF(g.x(), g.y()), button, button, Qt.KeyboardModifier.NoModifier))


def _move(w, local, gdelta):
    g = w.mapToGlobal(local) + gdelta
    w.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove, QPointF(local.x(), local.y()),
        QPointF(g.x(), g.y()), Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))


def _release(w, local, gdelta=QPoint(0, 0), button=Qt.MouseButton.LeftButton):
    g = w.mapToGlobal(local) + gdelta
    w.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(local.x(), local.y()),
        QPointF(g.x(), g.y()), button, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier))


def test_border_micro_drag_is_resize_not_click(app, tmp_path):
    """按在邊框上動 3px（小於 DRAG_THRESHOLD）必須算縮放，不能變成單擊。"""
    w = _Probe(Config(tmp_path / "c.json"))
    w.show()
    w0 = w.width()
    edge = QPoint(w.width() - 2, w.height() // 2)
    _press(w, edge)
    _move(w, edge, QPoint(3, 0))
    _release(w, edge, QPoint(3, 0))
    assert w.clicks == 0
    assert w.geometry_changes == 1
    assert w.width() == w0 + 3


def test_interior_micro_drag_is_still_a_click(app, tmp_path):
    """內部按下動 3px 仍視為單擊（既有行為，不得被上面的修正改壞）。"""
    w = _Probe(Config(tmp_path / "c.json"))
    w.show()
    center = QPoint(w.width() // 2, w.height() // 2)
    _press(w, center)
    _move(w, center, QPoint(3, 0))
    _release(w, center, QPoint(3, 0))
    assert w.clicks == 1
    assert w.geometry_changes == 0


def test_right_click_does_not_freeze_subtitle_countdown(app, tmp_path):
    """右鍵按下不得停掉字幕的自動隱藏倒數（以前對非左鍵完全不理會）。"""
    from app.ui.subtitle import SubtitleOverlay
    s = SubtitleOverlay(Config(tmp_path / "c.json"))
    s.show_result("中文", "English", None)
    assert s._hide_timer.isActive()
    center = QPoint(s.width() // 2, s.height() // 2)
    _press(s, center, Qt.MouseButton.RightButton)
    _release(s, center, button=Qt.MouseButton.RightButton)
    assert s._hide_timer.isActive()
