import threading

from pynput import keyboard, mouse

MOUSE_BUTTONS = {
    "middle": mouse.Button.middle,
    "x1": mouse.Button.x1,
    "x2": mouse.Button.x2,
}


def _resolve_key(name: str):
    name = name.lower().strip()
    special = getattr(keyboard.Key, name, None)
    if special is not None:
        return special
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise ValueError(f"無法識別的按鍵名稱：{name!r}")


class HotkeyListener:
    """全域監聽指定按鍵/滑鼠鍵的按住與放開。

    on_press_start / on_release_end 會在 pynput 的監聽執行緒被呼叫，
    呼叫端需自行轉回 UI 執行緒。
    """

    def __init__(self, on_press_start, on_release_end):
        self.on_press_start = on_press_start
        self.on_release_end = on_release_end
        self._active = False
        self._lock = threading.Lock()
        self._kb_listener = None
        self._mouse_listener = None
        self._target_key = None
        self._target_button = None

    def configure(self, hotkey_type: str, key_name: str):
        self.stop()
        self._active = False
        if hotkey_type == "mouse":
            self._target_button = MOUSE_BUTTONS.get(key_name.lower())
            if self._target_button is None:
                raise ValueError(f"不支援的滑鼠按鍵：{key_name!r}")
            self._target_key = None
            self._mouse_listener = mouse.Listener(on_click=self._on_click)
            self._mouse_listener.start()
        else:
            self._target_key = _resolve_key(key_name)
            self._target_button = None
            self._kb_listener = keyboard.Listener(
                on_press=self._on_key_press, on_release=self._on_key_release)
            self._kb_listener.start()

    def stop(self):
        if self._kb_listener is not None:
            self._kb_listener.stop()
            self._kb_listener = None
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None

    def _matches(self, key) -> bool:
        if key == self._target_key:
            return True
        # 一般字元鍵：KeyCode 比對 char（忽略大小寫）
        target_char = getattr(self._target_key, "char", None)
        pressed_char = getattr(key, "char", None)
        return (target_char is not None and pressed_char is not None
                and pressed_char.lower() == target_char.lower())

    # 回呼一律吞掉例外：例外會讓 pynput 的監聽執行緒整個停掉，
    # 熱鍵從此完全沒反應。

    def _on_key_press(self, key):
        try:
            if not self._matches(key):
                return
            with self._lock:
                if self._active:  # 按住時系統會重複觸發 press
                    return
                self._active = True
            self.on_press_start()
        except Exception:
            pass

    def _on_key_release(self, key):
        try:
            if not self._matches(key):
                return
            with self._lock:
                if not self._active:
                    return
                self._active = False
            self.on_release_end()
        except Exception:
            pass

    def _on_click(self, x, y, button, pressed):
        try:
            if button != self._target_button:
                return
            with self._lock:
                if pressed == self._active:
                    return
                self._active = pressed
            if pressed:
                self.on_press_start()
            else:
                self.on_release_end()
        except Exception:
            pass
