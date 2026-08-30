"""串流式系統字幕的純邏輯（本檔上半）與引擎（Task 4 補在下半）。

做法照 SakiRinn/LiveCaptions-Translator：只翻「最後一個句尾標點之後」的
目前句，文字穩定或夠長就翻、相同不重翻；完成句進歷史、疊加層顯示最近 N 行。
"""
import threading
import time

from dataclasses import dataclass, replace

from .local_translate import ModelLoadError

PUNC_EOS = ".?!。？！"
PUNC_COMMA = ",，、—\n"     # 顯示用的截斷點（照 TextUtil 的 PUNC_COMMA）
SHORT_THRESHOLD = 10       # 完成句短於此（UTF-8 位元組）→ 併進前一句
VERYLONG_THRESHOLD = 220   # 疊加層顯示原文的上限（UTF-8 位元組），超過就從頭截
WINDOW_MAX_SEC = 12.0      # 開放音訊窗上限
POLL_SEC = 0.5             # 兩輪辨識的最小間隔
IDLE_SEC = 1.25            # 文字停住這麼久就翻（= 它的 MaxIdleInterval 50 × 25ms）
MAX_SYNC = 3               # 文字連續變這麼多次（不算短句）就翻（= MaxSyncInterval）
SILENCE_COMMIT_SEC = 1.5   # 尾端靜音多久把未完句視為完成
DISPLAY_ROWS = 3           # 預設保留前幾句翻譯（rows 另外再帶當前句）
MIN_PEAK = 0.01            # 低於此視為靜音


@dataclass
class Sentence:
    text: str
    end: float          # 最後一個字的 end（秒，相對於送進辨識的音訊起點）
    start: float = 0.0  # 第一個字的 start（同一個時間軸）


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


def shorten_display_sentence(text: str, max_bytes: int = VERYLONG_THRESHOLD) -> str:
    """太長的句子只顯示後半段（照 TextUtil.ShortenDisplaySentence）。

    UTF-8 位元組數達 max_bytes 時，反覆把「第一個標點以前」砍掉，直到夠短
    或再也找不到標點（找不到就原樣回傳，寧可長也不要砍在半個詞上）。
    """
    while len(text.encode("utf-8")) >= max_bytes:
        index = next((i for i, ch in enumerate(text)
                      if ch in PUNC_EOS or ch in PUNC_COMMA), -1)
        if index < 0 or index + 1 >= len(text):
            break
        text = text[index + 1:].lstrip()
    return text


def split_sentences_by_words(words):
    """把帶時間戳的字切成 (完成句列表, 目前未完句或 None)。"""
    completed, pending = [], []
    for w in words:
        pending.append(w)
        if w.text and w.text[-1] in PUNC_EOS:
            completed.append(Sentence(join_words([p.text for p in pending]),
                                      pending[-1].end, pending[0].start))
            pending = []
    current = None
    if pending:
        current = Sentence(join_words([p.text for p in pending]),
                           pending[-1].end, pending[0].start)
    return completed, current


class CaptionState:
    """rows 與「要不要翻譯」的決策。無執行緒、無 IO，方便測試。

    GUI 執行緒只會呼叫 set_display_rows（單一屬性指派，CPython 的 GIL 下
    是原子操作，最壞情況只是晚一輪生效），其餘都在引擎執行緒上，
    因此刻意不加鎖，免得每輪辨識都要跟 GUI 搶鎖。
    """

    def __init__(self, display_rows=DISPLAY_ROWS):
        self.display_rows = display_rows
        self._finals = []
        self._current = None
        self._last_change_t = 0.0      # 目前句原文最後一次變動的時間（秒）
        self._sync_count = 0           # 原文變了幾次（短句不算），照 syncCount
        self._last_translated = None   # 上次送翻譯的 current 原文

    def set_display_rows(self, n: int):
        self.display_rows = max(1, int(n))

    @property
    def rows(self) -> list:
        """保留的前幾句 + 當前句：共 display_rows + 1 列。"""
        rows = list(self._finals)
        if self._current is not None:
            rows.append(self._current)
        return [replace(r) for r in rows[-(self.display_rows + 1):]]

    @property
    def previous_final_text(self) -> str:
        return self._finals[-1].original if self._finals else ""

    @property
    def has_current(self) -> bool:
        """還有未收的目前句（辨識器這輪沒回字時也要看得到）。"""
        return self._current is not None

    @property
    def current_row(self):
        """目前句的複本（沒有就 None）。

        引擎要在 commit_current() 之前先讀原文與既有翻譯：翻譯失敗時這句
        還留在目前句裡，下一輪可以整句重來。
        """
        return replace(self._current) if self._current is not None else None

    def update_current(self, text: str, now: float) -> bool:
        """更新目前句原文，回傳這一輪是否要翻譯它。

        觸發條件照 LiveCaptions-Translator 的 SyncLoop，只是把「輪數」換成
        「時間」（它每 25ms 一輪、我們每輪要跑一次 Whisper，輪數不可比）：
        - 原文停住 IDLE_SEC 沒變 → 翻（MaxIdleInterval）
        - 原文變了超過 MAX_SYNC 次（短句不計） → 翻並歸零（MaxSyncInterval）
        - 以句尾標點結尾 → 立刻翻並歸零
        相同文字不重翻（`_last_translated`），這點比它嚴格。
        """
        if self._current is None:
            self._current = Row(text)
            changed = True
        elif text != self._current.original:
            self._current.original = text   # 翻譯先留著，重翻後才換
            changed = True
        else:
            changed = False
        if changed:
            self._last_change_t = now
            if not self.is_short(text):
                self._sync_count += 1
        if not text or text == self._last_translated:
            return False
        if text[-1] in PUNC_EOS or self._sync_count > MAX_SYNC:
            self._sync_count = 0
            return True
        # 它的 idleCount 分支不歸零 syncCount，這裡照做
        return now - self._last_change_t >= IDLE_SEC

    def set_current_translation(self, translated: str):
        if self._current is not None:
            self._current.translated = translated
            self._last_translated = self._current.original

    @staticmethod
    def is_short(text: str) -> bool:
        """照 LiveCaptions-Translator 的 TextUtil：門檻算 UTF-8 位元組數。

        所以中日韓大約 3 個字以內算短句，英文則是 10 個字元以內。
        """
        return len(text.encode("utf-8")) < SHORT_THRESHOLD

    def commit_target(self, text: str) -> str:
        """這句收下去之後那一列的原文。

        短句會被併進前一句：合併後的整句才是要顯示、也才是要送去翻譯的
        文字（reference 的 Translator.cs 就是這樣接前一句的）。
        """
        prev = self.previous_final_text
        if prev and self.is_short(text):
            return join_words([prev, text])
        return text

    def commit_text(self, text: str) -> Row:
        """辨識器已判定完成的句子：進 finals，目前句清空。

        短句不另起一列，而是就地改寫前一列（回傳的就是那一列），不然前一句
        的翻譯會在畫面上出現兩次（自己一列、合併句又一列）。
        """
        target = self.commit_target(text)
        if self._finals and target != text:
            row = self._finals[-1]
            row.original = target
            row.translated = ""      # 呼叫端會補上合併後整句的翻譯
        else:
            row = Row(text, "", True)
            self._finals.append(row)
        self._reset_current()
        return row

    def commit_current(self):
        """把目前句直接當完成句（靜音／窗超長），保留已有翻譯。

        一樣套短句合併規則：短的尾句併進前一列，目前句本身就丟掉。
        """
        if self._current is None:
            return None
        target = self.commit_target(self._current.original)
        if self._finals and target != self._current.original:
            row = self._finals[-1]
            row.original = target
            row.translated = ""      # 呼叫端會補上合併後整句的翻譯
        else:
            row = self._current
            row.is_final = True
            self._finals.append(row)
        self._reset_current()
        return row

    def _reset_current(self):
        self._current = None
        self._last_change_t = 0.0
        self._sync_count = 0
        self._last_translated = None
        # 只保留顯示所需 + 一點餘裕，歷史另由 UI 記錄
        del self._finals[:-max(self.display_rows + 1, 5)]


class StreamingCaptionEngine:
    """每 POLL_SEC 一輪：辨識「上一句句尾之後」的音訊 → 切句 → 翻譯 → on_rows。

    所有回呼都在引擎執行緒上呼叫；stop() 之後一律不再回呼。
    now/sleep 可注入以便測試。
    """

    STT_WAIT_TIMEOUT = 300.0
    MIN_AUDIO_SEC = 0.5

    def __init__(self, buffer, stt, translator, languages, mic_busy,
                 on_rows, on_state, on_fatal, on_final,
                 display_rows=DISPLAY_ROWS, now=time.monotonic, sleep=time.sleep):
        self.buffer = buffer
        self.stt = stt
        self.translator = translator
        self._languages = languages
        self._mic_busy = mic_busy
        self._on_rows = on_rows
        self._on_state = on_state
        self._on_fatal = on_fatal
        self._on_final = on_final
        self._now = now
        self._sleep = sleep
        self.state = CaptionState(display_rows)
        self.committed_t = 0.0
        self._running = True
        self._thread = None
        self._stt_wait_started = None
        self._loading_announced = False

    # ---- 生命週期 ----

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return          # 連按兩次開始：不要多起一條執行緒
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="streaming-captions")
        self._thread.start()

    def stop(self):
        self._running = False
        thread, self._thread = self._thread, None
        # 致命錯誤的回呼是在引擎執行緒上發的，控制器會反手 stop() 引擎，
        # 這時 join 自己會丟 RuntimeError；此時只要把旗標放掉就好。
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.1)

    def set_display_rows(self, n: int):
        self.state.set_display_rows(n)

    def _run(self):
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
        while self._running:
            started = self._now()
            self.step()
            elapsed = self._now() - started
            remaining = POLL_SEC - elapsed
            # 分段睡，stop() 才能在 0.1 秒內收到
            while remaining > 0 and self._running:
                self._sleep(min(0.1, remaining))
                remaining -= 0.1

    # ---- 一輪 ----

    def step(self) -> bool:
        if not self._running:
            return False
        try:
            return self._step()
        except ModelLoadError as e:
            was_running = self._running
            self._running = False
            if was_running:     # 使用者已在翻譯途中 stop() → 不要再冒提示
                self._on_fatal(str(e))
            return False
        except Exception as e:
            if self._running:
                self._on_state("error", f"系統字幕處理失敗：{e}")
            return False

    def _step(self) -> bool:
        if self._mic_busy():
            return False   # 麥克風優先：使用者在說話時讓路
        if not self._wait_stt():
            return False
        audio = self.buffer.since(self.committed_t)
        sample_rate = self.buffer.sample_rate
        if len(audio) < sample_rate * self.MIN_AUDIO_SEC:
            return False

        # 整個開放窗都是靜音、又沒有未完句要收：跳過辨識，直接把窗推掉。
        # 不然持續靜音時窗會一路長到 WINDOW_MAX_SEC，每秒把同一段無聲音訊
        # 重算一次。有未完句時不能短路，否則靜音收句那條路就走不到了。
        open_sec = self.buffer.total_seconds - self.committed_t
        if (not self.state.has_current
                and self.buffer.tail_peak(open_sec) < MIN_PEAK):
            self.committed_t = self.buffer.total_seconds
            self.buffer.trim_before(self.committed_t)
            return False

        spoken, native = self._languages()
        base_t = self.committed_t
        words = self.stt.transcribe_words(audio, language=spoken, beam_size=1)
        if not self._running:
            return False
        completed, current = split_sentences_by_words(words)
        window_sec = self.buffer.total_seconds - base_t

        # 窗太長又沒有任何句尾：把目前這串強制當一句
        forced = not completed and current is not None and window_sec > WINDOW_MAX_SEC
        if forced:
            completed, current = [current], None

        for i, sentence in enumerate(completed):
            # 先翻譯、再 commit：翻譯丟例外時這一輪什麼都沒動（句子沒進
            # finals、committed_t 沒推進），同一段音訊下一輪會整句重來。
            # target 也要在 commit_text 之前取：短句會被併進前一列，合併後
            # 的整句才是這一列的原文，也才是要送去翻譯的文字；commit_text
            # 之後再問就會指到合併完的自己。
            target = self.state.commit_target(sentence.text)
            translated = self._translate(target, spoken, native)
            row = self.state.commit_text(sentence.text)
            row.translated = translated
            # 推進到「下一個還沒收的字」的起點，而不是這句最後一個字的
            # end：whisper 的 DTW 常把 end 抓得偏早，停在那裡的話句尾殘音
            # 還留在窗裡，下一輪會被辨識成一個碎片句（例如 "there."），
            # 變成重複的完成句與歷史列。
            if i + 1 < len(completed):
                next_start = completed[i + 1].start
            elif current is not None:
                next_start = current.start
            else:
                next_start = sentence.end
            # Whisper 偶爾給出重疊的字時間戳（下一個字的 start 早於這句的 end），
            # 不能讓 committed_t 倒退，否則這句會被再辨識、再 commit 一次
            self.committed_t = base_t + max(next_start, sentence.end)
            # 短句合併時歷史會同時留著「合併前那句」與「合併後的整句」，
            # 這是 reference 的紀錄行為，刻意不去回頭改寫上一筆歷史。
            if self._running:
                self._on_final(row.original, row.translated)
        if completed:
            if forced:
                # 強制收句後窗可能還是過長（句尾時間戳很早），committed_t
                # 要一併推進到窗上限內，否則下一輪會再收同一句。
                self.committed_t = max(self.committed_t,
                                       self.buffer.total_seconds - WINDOW_MAX_SEC)
            self.buffer.trim_before(self.committed_t)

        if (current is not None
                and self.state.update_current(current.text, self._now())):
            self.state.set_current_translation(
                self._translate(current.text, spoken, native))

        # 尾端安靜夠久 → 這句講完了。這裡刻意不看這輪有沒有辨識出 current：
        # faster-whisper 對幾乎全靜音的窗會回空 words，未完句還留在 state 裡，
        # 不在這收就永遠不會 final（會被下一句蓋掉）。
        if (self.state.has_current
                and self.buffer.tail_peak(SILENCE_COMMIT_SEC) < MIN_PEAK):
            self._commit_pending(spoken, native)
            self.committed_t = self.buffer.total_seconds
            self.buffer.trim_before(self.committed_t)
        elif current is None and not completed and window_sec > WINDOW_MAX_SEC:
            # 這段音訊辨識不出字（雜音、音樂）：太長就丟掉，別一直重算；
            # 丟之前先把未完句收掉，免得連它一起丟了。
            self._commit_pending(spoken, native)
            self.committed_t = self.buffer.total_seconds
            self.buffer.trim_before(self.committed_t)

        if self._running:
            self._on_rows(self.state.rows)
        return True

    # ---- 輔助 ----

    def _commit_pending(self, spoken: str, native: str):
        """把未收的目前句當完成句收掉（靜音、或要丟掉這段音訊之前）。

        一樣先翻譯、再 commit：翻譯丟例外時這句還是目前句，呼叫端的
        committed_t／trim 也跟著不會動，下一輪原封不動再收一次。
        """
        pending = self.state.current_row
        if pending is None:
            return
        # 併進前一列的話，這一列的原文變成合併後的整句，既有的翻譯就不能用了
        target = self.state.commit_target(pending.original)
        translated = pending.translated
        if target != pending.original or not translated:
            translated = self._translate(target, spoken, native)
        row = self.state.commit_current()
        row.translated = translated
        if self._running:
            self._on_final(row.original, row.translated)

    def _translate(self, text: str, spoken: str, native: str) -> str:
        announced = []

        def progress(message):
            if self._running:
                announced.append(True)
                self._on_state("loading", message)

        result = self.translator.translate(text, spoken, native,
                                           progress_cb=progress)
        if announced and self._running:
            self._on_state("listening", "正在聽系統聲音…")
        return result

    def _wait_stt(self) -> bool:
        """語音模型還在載入：回報一次「載入中」，這輪跳過；逾時視為致命。"""
        if self.stt.is_ready:
            self._stt_wait_started = None
            self._loading_announced = False
            return True
        now = self._now()
        if self._stt_wait_started is None:
            self._stt_wait_started = now
        if not self._loading_announced:
            self._loading_announced = True
            self._on_state("loading", "語音模型載入中，系統字幕稍後開始…")
        if now - self._stt_wait_started >= self.STT_WAIT_TIMEOUT:
            raise ModelLoadError("語音模型載入逾時，系統字幕已停止")
        return False
