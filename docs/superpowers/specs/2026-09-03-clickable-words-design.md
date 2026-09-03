# 麥克風字幕：單字方塊點擊接續朗讀、暫停、音量（設計）

日期：2026-09-03　狀態：已核准

## 需求（使用者原話）
麥克風收錄的聲音翻譯成英文後，每個單字變成可以點擊的方塊；點擊後從這個單字開始繼續往後念；字幕要多一個可以暫停的大按鈕；還有可以調整聲音大小的滑動（參考速度那條）。

## 範圍
只改麥克風字幕（`app/ui/subtitle.py`）與 TTS（`app/core/tts.py`）、設定頁「輸出」區、config。系統聲音字幕不動（它不朗讀）。

## 1. 單字方塊（英文行）
- `SubtitleOverlay` 的 `en_label`（QLabel）改成 `qfluentwidgets.FlowLayout` 容器裝一排 `WordChip`（QPushButton 或 QLabel+click），每字一塊：深色底 `rgba(255,255,255,26)`、圓角 6、hover 變亮 `rgba(255,255,255,60)`、白字、字級沿用現有英文行字級（`_font_sizes`/`_apply_fonts` 一併套用到方塊）。
- 切字：`english.split()`——標點黏著前字（"world." 一塊）。
- 點第 i 塊 → `tts.submit(" ".join(words[i:]), …)` 使用現有裝置/語速/音量設定；現有信箱機制自動打斷正在播的。點擊也算「重新朗讀」：倒數重算（沿用 `set_reading` 流程）。
- 佈局：FlowLayout 會自動換行；`_relayout`/`min_overlay_size`/`_reset_label_heights` 對英文行的量高改成量這個容器（`heightForWidth` 不適用 FlowLayout，用 `flow.heightForWidth(width)`——qfluentwidgets FlowLayout 有實作；若無，退而求其次呼叫容器 `sizeHint` 前先 `setGeometry`）。
- 中文行的就地編輯、其他行為不變。`show_message`（純訊息、無英文）時方塊區清空。

## 2. 暫停大按鈕
- 控制欄 🔊 上方加 48×48 的 `pause_button`（FluentIcon.PAUSE / PLAY_SOLID，Theme.DARK）。
- 播放中按 → `tts.pause()`：TTS worker 記下已播樣本數（用 play 開始的 `time.monotonic()` 差 × samplerate 估算）並 `sd.stop()`；圖示變 ▶。
- 再按 → `tts.resume()`：從記住的樣本位置 `sd.play(data[offset:], …)` 繼續；圖示變 ⏸。
- 暫停期間倒數維持暫停（現有 `set_reading(True)` 狀態不動；resume 播完才 `set_reading(False)`）。
- 新的 `submit`（新翻譯、重播、點字）清掉暫停狀態並把圖示復原成 ⏸。
- 沒在播放時按鈕 disabled（半透明）。

## 3. 音量滑桿
- `rate_slider` 下方加 `volume_slider`（同樣式、寬 92、範圍 0–100），標籤「音量 N%」。
- 即時寫 config `output.tts_volume`（預設 100）；播放時 `data * (volume/100)`（float32，clip 到 [-1,1]）。
- 拖動中播放不即時變（下一次 play/resume 生效即可，簡化）。
- 設定頁「輸出」區加同一個 SpinBox（0–100%），與字幕滑桿同步（照語速的既有雙向同步模式：`_sync_rate_from_config` / settings 訊號）。

## 4. TTS（`app/core/tts.py`）
- `_TTSWorker` 增加：`pause()`、`resume()`、`is_playing`/`is_paused`（執行緒安全，`_cond` 保護）；播放迴圈 `_wait_playback` 支援暫停（被 pause 停掉時不算 completed、也不算被新請求覆蓋，等 resume 或新請求）。
- resume 在 worker 執行緒執行（透過信箱送特殊「resume」請求或旗標——實作者決定，但所有 sd 呼叫必須留在 worker 執行緒，PortAudio 非執行緒安全）。
- `submit(..., volume: float = 1.0)`；模組層 `pause()/resume()/playback_state()` 轉發。
- 播放結束/被打斷的 `on_done` 語意不變；暫停不觸發 on_done。
- `tts_playing` 相關訊號（controller）沿用；暫停視為「仍在朗讀中」。

## 5. 測試
- tts：假 sd/soundfile/pyttsx3（現有測試已有假件模式，沿用）：volume 縮放有套用；pause 記錄 offset 且不觸發 on_done；resume 從 offset 播；pause 後新 submit 清掉暫停。
- subtitle（offscreen Qt）：`show_result` 產生 N 個方塊、標點黏前字；點第 i 塊呼叫的文字 == `" ".join(words[i:])`（monkeypatch tts.submit）；`show_message` 清空方塊；音量滑桿寫 config；暫停按鈕狀態切換（monkeypatch tts）。
- config 預設 `output.tts_volume == 100`。

## 不做
- 逐字朗讀高亮、系統字幕方塊化、播放中即時變音量。
