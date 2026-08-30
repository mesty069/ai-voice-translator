"""串流式系統字幕的純邏輯（本檔上半）與引擎（Task 4 補在下半）。

做法照 SakiRinn/LiveCaptions-Translator：只翻「最後一個句尾標點之後」的
目前句，文字穩定或夠長就翻、相同不重翻；完成句進歷史、疊加層顯示最近 N 行。
"""
from dataclasses import dataclass, replace

PUNC_EOS = ".?!。？！"
SHORT_THRESHOLD = 10       # 目前句短於此 → 翻譯時接前一句
MEDIUM_THRESHOLD = 40      # 文字變了且長於此 → 立即重翻
WINDOW_MAX_SEC = 12.0      # 開放音訊窗上限
POLL_SEC = 1.0             # 兩輪辨識的最小間隔
IDLE_ROUNDS = 2            # 文字連續幾輪沒變就翻
SILENCE_COMMIT_SEC = 1.5   # 尾端靜音多久把未完句視為完成
DISPLAY_ROWS = 3           # 預設顯示行數
MIN_PEAK = 0.01            # 低於此視為靜音


@dataclass
class Sentence:
    text: str
    end: float   # 最後一個字的 end（秒，相對於送進辨識的音訊起點）


@dataclass
class Row:
    original: str
    translated: str = ""
    is_final: bool = False


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF
            or 0xFF00 <= code <= 0xFFEF or 0x3000 <= code <= 0x303F)


def join_words(texts: list) -> str:
    """CJK 字之間不加空白，其餘字之間加一個空白。"""
    out = ""
    for text in texts:
        if not text:
            continue
        if out and not (_is_cjk(out[-1]) and _is_cjk(text[0])):
            out += " "
        out += text
    return out


def split_sentences_by_words(words):
    """把帶時間戳的字切成 (完成句列表, 目前未完句或 None)。"""
    completed, pending = [], []
    for w in words:
        pending.append(w)
        if w.text and w.text[-1] in PUNC_EOS:
            completed.append(Sentence(join_words([p.text for p in pending]),
                                      pending[-1].end))
            pending = []
    current = None
    if pending:
        current = Sentence(join_words([p.text for p in pending]), pending[-1].end)
    return completed, current


class CaptionState:
    """rows 與「要不要翻譯」的決策。無執行緒、無 IO，方便測試。"""

    def __init__(self, display_rows=DISPLAY_ROWS):
        self.display_rows = display_rows
        self._finals = []
        self._current = None
        self._idle = 0
        self._last_translated = None   # 上次送翻譯的 current 原文

    def set_display_rows(self, n: int):
        self.display_rows = max(1, int(n))

    @property
    def rows(self) -> list:
        rows = list(self._finals)
        if self._current is not None:
            rows.append(self._current)
        return [replace(r) for r in rows[-self.display_rows:]]

    @property
    def previous_final_text(self) -> str:
        return self._finals[-1].original if self._finals else ""

    def update_current(self, text: str) -> bool:
        """更新目前句原文，回傳這一輪是否要翻譯它。"""
        if self._current is None:
            self._current = Row(text)
            changed = True
            self._idle = 0
        elif text != self._current.original:
            self._current.original = text   # 翻譯先留著，重翻後才換
            changed = True
            self._idle = 0
        else:
            changed = False
            self._idle += 1
        if not text or text == self._last_translated:
            return False
        ends_sentence = text[-1] in PUNC_EOS
        return (ends_sentence or self._idle >= IDLE_ROUNDS
                or (changed and len(text) >= MEDIUM_THRESHOLD))

    def set_current_translation(self, translated: str):
        if self._current is not None:
            self._current.translated = translated
            self._last_translated = self._current.original

    def commit_text(self, text: str) -> Row:
        """辨識器已判定完成的句子：進 finals，目前句清空。"""
        row = Row(text, "", True)
        self._finals.append(row)
        self._reset_current()
        return row

    def commit_current(self):
        """把目前句直接當完成句（靜音／窗超長），保留已有翻譯。"""
        if self._current is None:
            return None
        row = self._current
        row.is_final = True
        self._finals.append(row)
        self._reset_current()
        return row

    def translate_input(self, text: str) -> str:
        prev = self.previous_final_text
        if prev and len(text) < SHORT_THRESHOLD:
            return f"{prev} {text}"
        return text

    def _reset_current(self):
        self._current = None
        self._idle = 0
        self._last_translated = None
        # 只保留顯示所需 + 一點餘裕，歷史另由 UI 記錄
        del self._finals[:-max(self.display_rows, 5)]
