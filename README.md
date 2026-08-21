# AI 語音中翻英

Windows 桌面工具：按住熱鍵說中文，放開後自動「語音辨識 → AI 梳理 → 翻譯成英文」。
錄音期間會暫時把其他程式的麥克風靜音（例如 Teams / Discord 對方會聽到無聲），
放開後自動恢復。

## 安裝

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 設定

第一次執行會自動產生 `config.json`（也可參考 `config.example.json`）。
必填：DeepSeek API key —— 可直接在程式的「設定」頁填入，或編輯 `config.json`：

```json
"ai": { "deepseek": { "api_key": "sk-..." } }
```

## 執行

```powershell
.venv\Scripts\python.exe -m app.main
```

首次啟動會下載 faster-whisper 語音模型（預設 small，約 460MB），
狀態列顯示「就緒」後即可使用。

## 使用方式

1. 按住 **F9**（可在設定改成其他鍵或滑鼠側鍵）開始說中文。
2. 放開後自動辨識、梳理、翻譯，主畫面顯示三欄結果，可一鍵複製英文。
3. 也可以直接在主畫面下方的輸入框打中文，按「翻譯輸入的文字」或 Ctrl+Enter。
4. 按視窗 X 或右上角縮小鈕會收成**懸浮球**（點一下還原視窗，熱鍵照常可用）；
   把球**拖到螢幕下方的 ✕ 關閉區**才會真正結束程式。
5. 選配功能（設定頁開關）：自動複製到剪貼簿、用 TTS 唸出英文（可選播放裝置）。

語音模型只會下載一次（存在 `~/.cache/huggingface`），之後啟動只是從硬碟載入。

## 架構重點

- AI 供應商採**策略模式 + 工廠**（`app/ai/`）：`TranslationProvider` 介面 +
  `DeepSeekProvider` 實作，`factory.py` 依 config 的 `provider` 欄位建立，
  之後要加 OpenAI / Claude 只需新增一個類別。
- 麥克風禁音（`app/core/mic_guard.py`）：透過 Windows Core Audio 列舉所有
  作用中錄音裝置的 session，除本程式外全部靜音並記住原狀態，放開後只恢復
  被本程式靜音的 session。
- 設計文件：`docs/superpowers/specs/2026-08-21-voice-translator-design.md`

## 測試

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```
