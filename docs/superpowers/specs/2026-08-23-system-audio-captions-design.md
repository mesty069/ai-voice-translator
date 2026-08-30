# 系統聲音即時雙語字幕 — 設計

日期：2026-08-23

## 目標

擷取電腦正在播放的聲音（YouTube、線上會議等，**不含自己的麥克風**），
即時轉成文字並翻譯成母語，以懸浮雙語字幕顯示。

翻譯**不走 DeepSeek**，改用本機模型，換取速度與零 API 成本。

## 已驗證的前提

| 項目 | 結果 |
|---|---|
| PortAudio/sounddevice 的 loopback | ❌ 無 loopback 或立體聲混音裝置 |
| `soundcard` 套件 WASAPI loopback | ✅ 實測擷取到系統聲音（峰值 0.34） |
| Argos Translate | ❌ 會拉進 torch+spacy+stanza（約 2–3GB），打包體積無法接受 |
| CTranslate2（已安裝，faster-whisper 的引擎） | ✅ 可直接跑翻譯模型，GPU 可用 |
| `transformers` tokenizer | ✅ 不會拉進 torch |
| 現成 CT2 翻譯模型 | ✅ `entai2965/nllb-200-distilled-600M-ctranslate2`、`gaudi/opus-mt-{src}-{tgt}-ctranslate2` |

## 資料流

```
系統輸出裝置（WASAPI loopback，soundcard）
  → 能量偵測切句（靜音 600ms 斷句；最長 8s 強制斷；最短 1s 才送）
  → faster-whisper 辨識（語言＝系統聲音語言，與麥克風共用模型並用鎖排隊）
  → 本機翻譯 CTranslate2（來源語言 → 母語）
  → caption_ready(原文, 譯文) 訊號
  → 系統聲音字幕浮層（雙語顯示 + 歷史）
```

## 元件

### `app/core/system_audio.py` — `SystemAudioCapture`
- 用 `soundcard` 對指定輸出裝置做 loopback 擷取，16kHz mono。
- 背景執行緒持續讀取，以能量門檻切句，透過 `on_segment(np.ndarray)` 回呼吐出。
- 介面：`start()` / `stop()` / `is_running`；建構時給裝置名稱與切句參數。
- 依賴：`soundcard`、`numpy`。不碰 UI、不碰 whisper。

### `app/core/local_translate.py` — `LocalTranslator`
- CTranslate2 推論 + `transformers.AutoTokenizer`。
- 模型登錄表（設定頁可切換）：
  - `nllb-600m` → `entai2965/nllb-200-distilled-600M-ctranslate2`（約 600MB，覆蓋全部語言）
  - `nllb-1.3b` → 同系列 1.3B（約 1.3GB，品質較好、較慢）
  - `opus-mt` → `gaudi/opus-mt-{src}-{tgt}-ctranslate2`（約 80MB／語言對）
- 首次使用才用 `huggingface_hub` 下載，過程回報進度給狀態列。
- 語言代碼映射：程式碼（zh/en/ja…）→ NLLB 代碼（eng_Latn/jpn_Jpan…）。
  **中文以 `zho_Hans` 解碼再用 OpenCC（s2twp）轉台灣繁體**：實測 NLLB-600M 用
  `zho_Hant` 會在逗號後系統性截斷（6 句中 3 句），簡體解碼則全部完整。
- `translate(text, src, tgt) -> str`；內部有鎖，GPU 不可用或 OOM 時自動退回 CPU。

### `app/core/system_captions.py` — `SystemCaptionsController`（QObject）
- 串接擷取 → 辨識 → 翻譯，全部在自己的工作執行緒，不佔用麥克風的 executor。
- 訊號：`caption_ready(str, str)`、`state_changed(str, str)`、`error_occurred(str)`。
- 與麥克風共用同一個 `SpeechToText`。`SpeechToText.transcribe` 加一把鎖，
  兩邊的辨識序列化，不會同時擠 GPU。
- 「麥克風優先」的實作：`AppController` 在錄音／處理期間把 `_session_active`
  設為 True，系統字幕在送出辨識前檢查此旗標，為 True 就先等待（最多 10 秒），
  避免使用者按住熱鍵說話時被系統字幕的辨識卡住。

### `app/ui/overlay_base.py` — `DraggableResizableOverlay`
- 從現有 `subtitle.py`（已 450 行）抽出共用行為：無邊框浮層、拖動移動、
  邊框/角落縮放、最小尺寸、位置大小記憶、透明度。
- `SubtitleOverlay`（麥克風）與 `SystemSubtitleOverlay` 都繼承它。
- 這是本次順帶的針對性重構，不擴大到無關的程式碼。

### `app/ui/system_subtitle.py` — `SystemSubtitleOverlay`
- 與麥克風字幕明顯區分：**青藍色系底 + 左上角「🔊 系統聲音」標籤**
  （麥克風字幕維持深灰底）。
- 內容：上行原文（中等字級）、下行母語譯文（大、亮）。
- 控制項：右上 ✕ 關閉、「歷史」按鈕展開本次整段逐字稿（可複製）。
- 持續顯示（不像麥克風字幕會倒數消失），停止擷取時才收起。

### 重疊推開（`main_window`）
- 任一字幕顯示或移動後檢查兩者矩形是否相交（含 12px 邊距）。
- 相交時把**後出現的那個**往上或往下推到不相交（選有空間的方向），帶動畫。
- 推開後的位置照常記憶。

## 設定（`config.json` 新增 `system_captions` 區段）

```json
"system_captions": {
  "enabled": false,
  "hotkey_type": "keyboard",
  "hotkey_key": "f11",
  "device": "default",        // loopback 來源輸出裝置
  "language": "",             // 空字串＝跟隨目標語言
  "engine": "nllb-600m",      // nllb-600m | nllb-1.3b | opus-mt
  "compute_device": "auto",   // auto | cpu
  "font_size": 20,
  "opacity": 100,
  "bg_color": "#0d2b33",
  "pos_x": null, "pos_y": null, "width": null, "height": null,
  "segment_silence_ms": 400,
  "max_segment_sec": 4
}
```

設定頁新增「系統聲音字幕」區塊：啟用開關、熱鍵、擷取裝置、聲音語言、
翻譯模型、翻譯裝置、字級、透明度。

開關三處同一狀態（沿用浮動輸入框的既有模式）：熱鍵、設定頁、懸浮球長按選單。

## 錯誤處理

| 情況 | 行為 |
|---|---|
| `soundcard` 不可用／擷取裝置消失 | 顯示錯誤、自動停止、開關回到關閉 |
| 翻譯模型下載失敗 | 顯示錯誤並停止；不影響麥克風翻譯 |
| GPU 記憶體不足 | 翻譯自動退回 CPU 並提示一次 |
| 辨識結果為空或全靜音 | 略過該段，不顯示字幕 |
| 麥克風錄音同時進行 | 共用語音模型，用鎖排隊，麥克風優先 |

## 測試

- **單元**：切句邏輯（合成音訊：靜音/語音/超長段）、語言代碼映射、
  重疊推開的幾何計算。
- **整合（離屏）**：字幕浮層顯示/拖動/縮放/推開；以 stub 翻譯器跑完整 pipeline。
- **實機**：播放 YouTube 實測端到端延遲與字幕正確性。

## 風險

1. **延遲**：辨識＋翻譯約需 0.5–1 秒，字幕會落後語音約一句。這是切句式即時
   字幕的固有特性，非缺陷。
2. **顯示記憶體**：8GB 顯卡已被其他軟體佔用約 3.4GB；語音模型約 1.6GB，
   翻譯模型再 0.6–1.3GB 會吃緊 → 提供「翻譯跑 CPU」設定。
3. **翻譯品質**：本機模型明顯不如 DeepSeek，這是速度與成本的取捨；
   模型可在設定頁切換以便比較。

## 不做（YAGNI）

- 不做喇叭聲與麥克風聲的混合辨識（明確排除自身麥克風）。
- 不做講者分離（diarization）。
- 不做字幕匯出檔案（歷史可複製即可）。
- 不做兩段式「先小模型再補正」的改寫流程。
