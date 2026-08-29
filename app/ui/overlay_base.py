from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtWidgets import QApplication, QWidget

_MAX = 16777215


class DraggableResizableOverlay(QWidget):
    """無邊框浮層的共用行為：拖動移動、邊框/角落縮放、位置大小記憶、透明度。

    子類別必須設定 CONFIG_SECTION（config.json 中的區段名），
    位置與大小會存進該區段的 pos_x / pos_y / width / height。
    """

    CONFIG_SECTION = ""
    BORDER = 8            # 邊框拖曳調整大小的感應寬度
    DRAG_THRESHOLD = 6

    def __init__(self, config, flags, parent=None):
        super().__init__(parent, flags)
        self.config = config
        self.setMouseTracking(True)
        self._pressed = False
        self._moved = False
        self._resize_edges = None
        self._press_geo = None
        self._press_global = QPoint()
        self._drag_offset = QPoint()

    # ---- 幾何記憶 ----

    def min_overlay_size(self) -> QSize:
        return QSize(240, 100)

    def saved_pos(self):
        x = self.config.get(self.CONFIG_SECTION, "pos_x", default=None)
        y = self.config.get(self.CONFIG_SECTION, "pos_y", default=None)
        if x is None or y is None:
            return None
        pos = QPoint(int(x), int(y))
        # 儲存的位置必須還在某個螢幕上（拔掉外接螢幕後回到預設位置）
        if QApplication.screenAt(pos) is None:
            return None
        return pos

    def saved_size(self):
        w = self.config.get(self.CONFIG_SECTION, "width", default=None)
        h = self.config.get(self.CONFIG_SECTION, "height", default=None)
        if not w or not h:
            return None
        return int(w), int(h)

    def save_geometry(self):
        self.config.set(self.CONFIG_SECTION, "pos_x", self.pos().x())
        self.config.set(self.CONFIG_SECTION, "pos_y", self.pos().y())
        self.config.set(self.CONFIG_SECTION, "width", self.width())
        self.config.set(self.CONFIG_SECTION, "height", self.height())

    def target_opacity(self) -> float:
        percent = self.config.get(self.CONFIG_SECTION, "opacity", default=100)
        return max(0.3, min(1.0, int(percent) / 100))

    def apply_opacity(self):
        if self.isVisible():
            self.setWindowOpacity(self.target_opacity())

    # ---- 覆寫點 ----

    def _on_simple_click(self, pos):
        """單擊（沒有拖動）時呼叫。"""

    def _on_geometry_changed(self):
        """拖動或縮放結束時呼叫。"""
        self.save_geometry()

    # ---- 滑鼠 ----

    def _edges_at(self, pos):
        left = pos.x() <= self.BORDER
        right = pos.x() >= self.width() - self.BORDER
        top = pos.y() <= self.BORDER
        bottom = pos.y() >= self.height() - self.BORDER
        if left or right or top or bottom:
            return (left, right, top, bottom)
        return None

    def _cursor_for(self, edges):
        left, right, top, bottom = edges
        if (left and top) or (right and bottom):
            return Qt.CursorShape.SizeFDiagCursor
        if (left and bottom) or (right and top):
            return Qt.CursorShape.SizeBDiagCursor
        if left or right:
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._pressed = True
        self._moved = False
        self._press_global = event.globalPosition().toPoint()
        self._press_geo = self.geometry()
        self._resize_edges = self._edges_at(event.position().toPoint())
        self._drag_offset = self._press_global - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        gpos = event.globalPosition().toPoint()
        if not self._pressed:
            edges = self._edges_at(pos)
            self.setCursor(self._cursor_for(edges) if edges
                           else Qt.CursorShape.ArrowCursor)
            return
        if not self._moved:
            if (gpos - self._press_global).manhattanLength() < self.DRAG_THRESHOLD:
                return
            self._moved = True
        if self._resize_edges:
            self._apply_resize(gpos)
        else:
            self.move(gpos - self._drag_offset)

    def _apply_resize(self, gpos):
        left, right, top, bottom = self._resize_edges
        geo = self._press_geo
        dx = gpos.x() - self._press_global.x()
        dy = gpos.y() - self._press_global.y()
        min_size = self.minimumSize()
        x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()
        if right:
            w = max(min_size.width(), geo.width() + dx)
        if bottom:
            h = max(min_size.height(), geo.height() + dy)
        if left:
            w = max(min_size.width(), geo.width() - dx)
            x = geo.x() + geo.width() - w
        if top:
            h = max(min_size.height(), geo.height() - dy)
            y = geo.y() + geo.height() - h
        self.setGeometry(x, y, w, h)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self._pressed:
            return
        self._pressed = False
        self._resize_edges = None
        if self._moved:
            self._on_geometry_changed()
        else:
            self._on_simple_click(event.position().toPoint())
