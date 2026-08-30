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
