# 系統聲音字幕：串流式三行字幕（Live Captions 風格）設計

日期：2026-08-30
狀態：已核准（使用者選擇「照 LiveCaptions-Translator 的做法」）

## 目標

把系統聲音字幕從「切段 → 辨識 → 翻譯 → 顯示一句」改成 Live Captions 風格：

- 字幕**邊講邊長出來**：目前這句的原文每秒更新，翻譯在文字穩定時更新、文字變了就重翻。
- 句子**不再被固定秒數硬切**：完成的句子以句號／問號／驚嘆號判定，音訊起點跟著句尾推進。
- 疊加層固定顯示 **3 行**（可設定 1–5）：最近 N-1 句完成的 + 最下面 1 句正在長的；新句進來舊句往上頂掉。
- 翻譯時把顯示中的幾行原文一起送批次；太短的句子接前一句一起翻（`SHORT_THRESHOLD`）。

參考實作：SakiRinn/LiveCaptions-Translator `src/Translator.cs`、`src/utils/TextUtil.cs`、`src/models/Caption.cs`。

## 與參考實作的關鍵差異

Windows Live Captions 是串流辨識器，文字自己會長；Whisper 不是。我們用**重複辨識開放音訊窗**模擬：
每約 1 秒把「上一個完成句子的句尾時間點」之後的音訊（上限 `WINDOW_MAX_SEC`）整段重新辨識，
得到一段會越來越長的文字。faster-whisper 開 `word_timestamps=True`，句尾字的 `end` 時間戳用來推進音訊起點。

## 架構

```
SystemAudioCapture(loopback)         ── 連續音訊 frames ──▶ RollingAudioBuffer
                                                                │
StreamingCaptionEngine（worker 執行緒，每 POLL_SEC 一輪）        │
  1. audio = buffer.since(committed_t)                          ◀┘
  2. words = stt.transcribe_words(audio, language, beam_size=1)     # 帶時間戳
  3. text = join(words)；以 PUNC_EOS 切成 completed[] + current
  4. 對每個 completed 句：commit → 推進 committed_t 到句尾字 end；送翻譯；進 rows
  5. current：與上輪比較 → 變了：更新原文、重設 idle；沒變：idle+1
     觸發翻譯：current 以 EOS 結尾 ∨ idle ≥ IDLE_ROUNDS ∨ (變了且 len ≥ MEDIUM_THRESHOLD)
     相同文字不重翻（上次翻過的 current 原文快取）
  6. 靜音：buffer 尾端 SILENCE_COMMIT_SEC 內峰值 < min_peak 且 current 非空 → 視為完成句
  7. emit rows_changed(rows)   # rows = 最近 DISPLAY_ROWS 個 (original, translated, is_final)
```

- `RollingAudioBuffer`（新，`app/core/system_audio.py`）：thread-safe，`append(frames)`、`since(t_sec) -> np.ndarray`、`trim_before(t_sec)`、`duration`、`tail_peak(sec)`。以絕對樣本數計時間；`committed_t` 為絕對秒。
- `SystemAudioCapture`：改為 `on_frames(np.ndarray)` 回呼（每 0.1 秒一塊），不再切段；`SegmentAccumulator` 刪除（含測試）。`on_error` 不變。
- `SpeechToText.transcribe_words(audio, language, beam_size) -> list[Word]`（新）：`Word(text, start, end)`；`transcribe` 維持不變供麥克風用。鎖內執行，與 `transcribe` 互斥。
- `StreamingCaptionEngine`（新，`app/core/streaming_captions.py`）：純邏輯 + 執行緒，不含 Qt；透過回呼 `on_rows(list[Row])`、`on_state(state, msg)`、`on_fatal(msg)` 對外。`Row = (original: str, translated: str, is_final: bool)`。
- `SystemCaptionsController`：改為組裝 capture + buffer + engine + translator，把回呼轉成 Qt 訊號 `rows_changed(list)`；`source_ready`/`caption_ready` 刪除。世代編號機制保留（engine 每次 start 新建，stop 後舊 engine 回呼一律忽略）。
- `SystemSubtitleOverlay.set_rows(rows)`：畫 N 行，每行原文小字＋翻譯大字；未完成行的翻譯欄空白時顯示「…」；歷史面板仍記錄所有完成句。

## 常數（沿用參考實作，位於 `streaming_captions.py`）

| 名稱 | 值 | 意義 |
|---|---|---|
| `PUNC_EOS` | `".?!。？！"` | 句尾 |
| `SHORT_THRESHOLD` | 10 字元 | 目前句短於此 → 翻譯時接前一句 |
| `MEDIUM_THRESHOLD` | 40 字元 | 文字變了且長於此 → 立即重翻 |
| `WINDOW_MAX_SEC` | 12.0 | 開放音訊窗上限；超過則把最早的字強制當句尾 commit |
| `POLL_SEC` | 1.0 | 兩輪辨識的最小間隔（上一輪沒跑完不開新的） |
| `IDLE_ROUNDS` | 2 | 文字連續幾輪沒變就翻譯（≈2 秒，對應參考實作 1.25 秒） |
| `SILENCE_COMMIT_SEC` | 1.5 | 尾端靜音多久把未完句視為完成 |
| `DISPLAY_ROWS` | 3 | 預設顯示行數，config `system_captions.display_rows`，範圍 1–5 |

## 翻譯策略

- 完成句：`translator.translate(text)`，若 `len(text) < SHORT_THRESHOLD` 且有前一句 → 翻 `prev + " " + text`，只取結果中對應 text 的部分做不到（NLLB 句子級），因此直接把合併結果顯示在**目前行**、前一行翻譯不變。
- 目前句：同上規則；同一原文不重翻；翻譯在 engine 執行緒同步進行（翻譯 0.1–0.3 秒，可接受），但若翻譯回來時 current 已改變，結果仍顯示（下一輪會再翻）。
- 誠實說明：NLLB／OPUS-MT 為句子級模型，「三行一起送」只是一次批次，不會產生跨句上下文；品質提升來自句子不再被硬切。

## 設定頁

- 移除「每段最長」「停頓多久算一句」。
- 新增「顯示行數」SpinBox 1–5（`display_rows`），變更即時生效（不重啟管線）。
- 保留：來源裝置、語言、模型、運算裝置、字體、透明度。

## 錯誤處理

- 辨識或翻譯單輪失敗 → `on_state("error", msg)`，下一輪繼續（暫時錯誤）。
- `ModelLoadError`、擷取死亡 → `on_fatal` → 關閉功能（現有行為）。
- GPU 被占：一輪超過 `POLL_SEC` 就自然拉長間隔，不堆積。
- 麥克風翻譯進行中（`mic_busy()`）：本輪跳過辨識（讓路），維持現有邏輯。

## 測試

- `RollingAudioBuffer`：append/since/trim/tail_peak 正確；跨 trim 的絕對時間不漂移。
- `StreamingCaptionEngine`（stt/translator 皆為假物件，可注入 `now()`）：
  - 文字長出來時 rows 的目前行原文更新、翻譯不變；idle 2 輪後翻譯出現
  - 相同 current 不重翻（translator 呼叫次數）
  - EOS 出現 → 句子 commit、committed_t 推進到句尾字 end、rows 上推、最多 DISPLAY_ROWS
  - 短句接前一句翻譯
  - 尾端靜音 1.5 秒 commit
  - WINDOW_MAX_SEC 超過強制 commit
  - stop 後舊回呼不觸發
- `SpeechToText.transcribe_words`：假模型回傳 words 結構被正確轉換；與 `transcribe` 共用鎖
- overlay：`set_rows` 行數與內容
- 現有 config 測試更新（`display_rows` 預設 3；移除 segment 兩鍵）

## 不做

- 不改麥克風翻譯管線與其字幕。
- 不引入 torch／任何新的重量級依賴。
- 不做 LLM 上下文翻譯（DeepSeek 不用於系統字幕，維持原決定）。

## 2026-08-30 對齊 LiveCaptions-Translator（除翻譯引擎）

參考 `src/Translator.cs`（SyncLoop）、`src/windows/OverlayWindow.xaml`、`src/utils/TextUtil.cs`。

### 觸發時序（`app/core/streaming_captions.py`）

它每 25ms 讀一次系統字幕，所以用「輪數」計時；我們每一輪要跑一次 Whisper
（0.3–0.8 秒），輪數不可比，因此把 idle 改成用**時間**判斷，其餘照搬：

- 原文停住 `IDLE_SEC` 沒變 → 翻（對應它的 `MaxIdleInterval` 50 × 25ms = 1.25 秒）。
- 原文變了且不算短句（UTF-8 位元組 ≥ `SHORT_THRESHOLD`）→ `_sync_count += 1`；
  `_sync_count > MAX_SYNC` → 翻並歸零（對應 `MaxSyncInterval`）。
- 以句尾標點結尾 → 立刻翻並歸零。
- idle 這條**不**歸零 `_sync_count`（照它的寫法）。
- 相同原文不重翻（`_last_translated`），這點比它嚴格，刻意保留。
- `CaptionState.update_current(text, now)` 多接一個時間參數，引擎傳 `self._now()`，
  測試可注入可推進的假時鐘。

### 新的常數表

| 名稱 | 值 | 意義 |
|---|---|---|
| `POLL_SEC` | 0.5 | 兩輪辨識的最小間隔（上一輪沒跑完不開新的） |
| `IDLE_SEC` | 1.25 | 原文停住這麼久就翻（取代 `IDLE_ROUNDS`） |
| `MAX_SYNC` | 3 | 原文變動次數超過此值就翻並歸零（短句不計） |
| `VERYLONG_THRESHOLD` | 220（UTF-8 位元組） | 疊加層顯示原文的上限，超過就從頭截 |
| `PUNC_COMMA` | `",，、—\n"` | `shorten_display_sentence` 的截斷點 |
| `SHORT_THRESHOLD` | 10（UTF-8 位元組） | 不變：短的完成句併進前一列；短句不計 sync |
| `DISPLAY_ROWS` | 3 | 改為「保留前幾句翻譯」，`rows` 回傳 `display_rows + 1` 列 |

- 刪除 `MEDIUM_THRESHOLD`（它只用在 `ReplaceNewlines` 的顯示換行，我們不需要）。
- 新增 `shorten_display_sentence(text, max_bytes)`：照 `TextUtil.ShortenDisplaySentence`，
  位元組數達上限時反覆砍掉「第一個標點以前」，找不到標點就原樣回傳。

### 疊加層版面（`app/ui/system_subtitle.py`）

照 `OverlayWindow.xaml` 改成兩個 QLabel（`row_widgets` 移除）：

- 上方 `translation_label`：把保留的前幾句與當前句的翻譯用 `join_words` 接成
  **一段**，前幾句 `#d0d0d0`、當前句白色，以 rich text `<span style="color:…">`
  上色（文字先 `html.escape`）。當前句還沒翻好時就只顯示前幾句，不補「…」。
- 下方 `original_label`：字級 `round(font_size * 0.8)`、顏色 `#cfe9ef`，顯示
  `shorten_display_sentence(rows[-1].original, VERYLONG_THRESHOLD)`。
- `rows[-1]` 不論 `is_final` 都當「當前句」（句子講完後仍留在畫面上，直到新字出現）。
- `rows` 為空 → 兩個 label 都清空。`required_height`/`_fit_height`/`resizeEvent`
  邏輯不變，只是改量這兩個 label。歷史面板不變。
- 設定頁文案改為「保留前幾句翻譯（它們會接在正在講這句的前面）」，單位「句」；
  範圍 1–5 與 config key `display_rows` 不變。

### 不做

- DisplayLoop 的 720ms 視覺停頓、Log Cards/CSV、翻譯引擎與麥克風管線都不動。
