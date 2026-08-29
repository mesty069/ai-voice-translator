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
