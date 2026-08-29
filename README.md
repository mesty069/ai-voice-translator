# AI 語音中翻英

Windows 桌面工具：按住熱鍵說中文，放開後自動「語音辨識 → AI 梳理 → 翻譯成英文」，
並自動朗讀英文。支援懸浮球模式：字幕顯示結果、浮動輸入框打字翻譯。

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

- 雙擊專案根目錄的 **`AI語音中翻英.exe`**（會確保單一實例；重複開啟會叫出現有視窗），或：

```powershell
.venv\Scripts\python.exe -m app.main
```

- **釘選到工作列**：跑一次 `tools/create_shortcut.py` 會在開始功能表建立捷徑
  （含 AppUserModelID，執行中的視窗會與釘選圖示合併），再從開始功能表右鍵釘選。
- **開機自動啟動**：設定頁「一般」區開關（背景預載模型，用時零等待）。

首次啟動會下載 faster-whisper 語音模型（預設 large-v3，約 3GB，只下載一次，
存在 `~/.cache/huggingface`），狀態列顯示「就緒」後即可使用。
預設用 GPU（CUDA，int8_float16 量化），無 GPU 的機器自動退回 CPU。

## 使用方式

### 語音翻譯
1. 按住**錄音熱鍵**（預設 F9，設定頁點擊後按任意鍵/滑鼠鍵即可換）說中文。
2. 錄音時狀態列顯示**即時音量峰值**；放開後自動辨識、梳理、翻譯、**自動朗讀英文**。
3. 錄音期間會**隔離麥克風**：把系統預設錄音裝置切到已靜音的誘餌裝置，
   跟隨系統預設的軟體（Discord/Teams 等）聽不到你說話；放開後恢復。

### 文字翻譯
- 主畫面輸入框打中文，**Enter 送出**（Ctrl+Enter 換行）。
- 英文翻譯區有 🔊 朗讀與複製按鈕。

### 懸浮球模式
- 按**縮小鈕**或**視窗被切到背景**（可關）→ 收成懸浮球；按 **X** 或把球拖到
  螢幕下方 ✕ 區、或長按選單「結束程式」→ 真正退出。
- 球可拖到任何位置；**點一下**還原視窗；**長按**展開選單（輸入框開關/主視窗/結束）。
- 翻譯結果以**字幕**顯示：可拖動、邊框調大小（尊重你設的大小，字放不下自動縮小）、
  右上 ✕ 關閉、🔊 重播、語速拖桿；**點中文行可直接改字**，Enter 或點出去重新翻譯；
  字幕顯示中**直接按 Enter** 開空白輸入打新句子。
- **浮動輸入框**（長按球或設定頁開啟）：可拖、可調大小、Enter 送出。
- 字幕外觀可調：顯示秒數、字體大小/顏色/粗細/字型、底色、透明度。

### 朗讀
- 翻譯完成自動朗讀（可關）；語速可在設定頁或字幕拖桿調整。
- 播放為單一通道：新的朗讀（含 🔊 重播、朗讀熱鍵）會**打斷並取代**正在播的。
- **朗讀熱鍵**（選配）：字幕顯示時按了重播；字幕消失後無效。

## 架構重點

- AI 供應商採**策略模式 + 工廠**（`app/ai/`）：`TranslationProvider` 介面 +
  `DeepSeekProvider` 實作，`factory.py` 依 config 的 `provider` 欄位建立，
  之後要加 OpenAI / Claude 只需新增一個類別。
- 錄音走 **WASAPI**（`app/core/recorder.py`）：MME 在系統預設裝置改變時會重排
  裝置編號（麥克風隔離功能正是靠切換預設裝置），WASAPI 裝置識別穩定不受影響。
- 麥克風隔離（`app/core/mic_isolation.py`）：切換預設錄音裝置到誘餌＋靜音其他
  錄音裝置端點。注意：Windows 對錄音裝置的 per-app session 靜音等於整顆裝置
  靜音（實測），所以「只靜音同裝置上的其他 app」做不到；手動綁定同一顆實體
  麥克風的軟體無法隔離。
- TLS 用 **truststore**（Windows 憑證存放區）：防毒（如 Norton Web Shield）會
  攔截 HTTPS 重簽憑證，Python 內建 certifi 清單會驗證失敗。
- TTS 可打斷（`app/core/tts.py`）：序號機制，後來的請求作廢/覆蓋前面的播放；
  在獨立執行緒執行，不占用翻譯流程的工作佇列。
- 設計文件：`docs/superpowers/specs/2026-08-21-voice-translator-design.md`

## 測試

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```
