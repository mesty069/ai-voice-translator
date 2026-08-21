# 語音中翻英工具（Windows / Python）設計文件

日期：2026-08-21
狀態：已與使用者確認（STT 由 Windows 內建改為 faster-whisper，使用者於實作前指定）

## 目的

Windows 桌面工具：按住熱鍵（鍵盤或滑鼠）說中文，放開後自動完成
「語音轉文字 → AI 梳理修正 → 翻譯成英文」，過程中暫時把其他軟體的
麥克風靜音，避免通話對方聽到使用者對工具說的中文。

## 整體流程

```
按住熱鍵
  ├─ 靜音其他 app 的麥克風錄音 session（記住原狀態）
  └─ 開始錄音（sounddevice，16kHz mono）
放開熱鍵
  ├─ 停止錄音
  ├─ 恢復其他 app 麥克風（try/finally，無論後續成敗都恢復）
  └─ 背景 pipeline：
       faster-whisper 轉中文文字
       → AI 一次呼叫完成「梳理 + 翻譯」（回 JSON：refined、english）
       → UI 顯示三欄：原始辨識 / 梳理後中文 / 英文翻譯
       → 依設定：自動複製剪貼簿、TTS 唸英文（可選輸出裝置）
```

## 技術選型

| 元件 | 選擇 | 備註 |
|---|---|---|
| UI | PySide6 + PySide6-Fluent-Widgets | Windows 11 Fluent 風格，內建深/淺色主題與主題色切換 |
| STT | faster-whisper（本地） | 無 NVIDIA GPU → CPU int8；模型大小可在設定切換（預設 small，首次執行自動下載）；language="zh" |
| 錄音 | sounddevice | 16kHz mono float32，直接餵 whisper |
| 全域熱鍵 | pynput | 支援鍵盤按住與滑鼠側鍵按住，全域監聽 |
| 禁音其他 app | pycaw（Core Audio） | 列舉所有作用中錄音裝置的 session，除本程式外全部靜音並記住原狀態，放開後恢復 |
| AI 翻譯 | 策略模式 + 工廠 | `TranslationProvider` 抽象介面；第一個實作 DeepSeek（OpenAI 相容 API）；工廠依 config `provider` 欄位建立 |
| TTS | pyttsx3（SAPI5）產生 wav + sounddevice 播放 | 可選輸出裝置 |

## 設定檔（config.json，gitignore；提供 config.example.json）

```json
{
  "ai": {
    "provider": "deepseek",
    "deepseek": { "api_key": "", "base_url": "https://api.deepseek.com", "model": "deepseek-chat" }
  },
  "stt": { "model_size": "small", "device": "cpu", "compute_type": "int8" },
  "hotkey": { "type": "keyboard", "key": "f9" },
  "output": { "auto_copy": false, "tts_enabled": false, "tts_device": "default" },
  "mute_other_apps": true,
  "ui": { "theme": "auto", "theme_color": "#0078d4" }
}
```

UI 設定頁可視化編輯所有欄位並即時存檔。

## 專案結構

```
app/
  main.py            # 進入點，組裝各元件
  config.py          # 設定讀寫 + 預設值合併
  core/
    recorder.py      # sounddevice 錄音
    stt.py           # faster-whisper 轉文字（lazy load，背景載入）
    hotkey.py        # pynput 按住/放開監聽
    mic_guard.py     # pycaw 禁音/恢復其他 app
    tts.py           # 英文語音播放（可選裝置）
  ai/
    base.py          # TranslationProvider 抽象類
    deepseek.py      # DeepSeek 實作
    factory.py       # 依 config 建 provider
  ui/
    main_window.py   # 主視窗（狀態指示、三欄結果、複製）
    settings_page.py # 設定頁
tests/               # config、factory、deepseek 解析的單元測試
config.example.json
requirements.txt
```

## 錯誤處理

- 放開熱鍵時無論如何先恢復麥克風（try/finally）。
- 按住 < 0.3 秒視為誤觸，忽略。
- API key 未填或呼叫失敗：顯示錯誤 + 保留原始辨識中文。
- whisper 模型載入/下載中：狀態列提示，期間熱鍵無效。
- 錄到的音訊過短或全靜音：提示「沒有收到聲音」。

## 執行緒模型

- Qt 主執行緒：UI。
- pynput 監聽執行緒：按下/放開事件 → 轉發（Qt queued signal）。
- 按下：worker 執行緒做 mute + 開錄音；放開：stop + unmute + pipeline
 （STT → AI）跑在單一背景 worker，完成後以 signal 回 UI。
- comtypes COM 操作在 worker 內以 CoInitialize/CoUninitialize 包裹。

## 驗收

- 單元測試：config 讀寫合併、factory 建立、DeepSeek 回應解析（mock HTTP）。
- 實跑：app 可啟動、錄音→轉文字→翻譯全流程實測。
- 禁音效果：需使用者以 Teams/Discord 等實測確認。
