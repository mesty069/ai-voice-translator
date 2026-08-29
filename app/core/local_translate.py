import re
import threading

from .cuda_dlls import register_cuda_dll_dirs

# 程式的語言代碼 → NLLB-200 的語言代碼
# 中文刻意用簡體 zho_Hans 解碼再以 OpenCC 轉台灣繁體：NLLB 的繁體訓練資料
# 遠少於簡體，直接用 zho_Hant 會系統性地在逗號後截斷（實測 6 句中 3 句），
# 換 zho_Hans 全部完整。
NLLB_CODES = {
    "zh": "zho_Hans",
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "ru": "rus_Cyrl",
}

# 已驗證存在且內含 tokenizer 的 CTranslate2 模型
ENGINES = {
    "nllb-600m": {
        "kind": "nllb",
        "repo": "entai2965/nllb-200-distilled-600M-ctranslate2",
    },
    "nllb-1.3b": {
        "kind": "nllb",
        "repo": "entai2965/nllb-200-distilled-1.3B-ctranslate2",
    },
    "opus-mt": {
        "kind": "opus",
        "repo": "gaudi/opus-mt-{src}-{tgt}-ctranslate2",
    },
}

ENGINE_LABELS = [
    ("nllb-600m", "NLLB 600M（約 600MB，通用、快）"),
    ("nllb-1.3b", "NLLB 1.3B（約 1.3GB，品質最好、較慢）"),
    ("opus-mt", "OPUS-MT（約 80MB，單一語言對、最快）"),
]


class ModelLoadError(RuntimeError):
    """翻譯模型下載或載入失敗（網路、tokenizer、推論引擎皆算）。

    呼叫端（system_captions.py）以此類別（而非訊息字串前綴）判斷是否為
    致命錯誤：一旦模型載不起來，之後每一段都會一樣失敗，必須停止功能，
    而不是把它當成單段音訊的暫時性錯誤，導致每段都重新嘗試下載。
    """


_opencc = None


def to_traditional(text: str) -> str:
    """簡體 → 台灣繁體（含用語轉換，如 服务器→伺服器）。"""
    global _opencc
    if _opencc is None:
        from opencc import OpenCC
        _opencc = OpenCC("s2twp")
    return _opencc.convert(text)


# CJK 字元範圍：中日韓表意文字＋平假名／片假名＋諺文音節，用來判斷
# 「這個半形標點是不是緊貼著中日韓文字」。範圍邊界字元已用工具驗證過
# code point 正確（U+4E00-U+9FFF、U+3400-U+4DBF、U+F900-U+FAFF、
# U+3040-U+309F、U+30A0-U+30FF、U+AC00-U+D7AF），複製/編輯這段時
# 務必用同樣的方式重新驗證，避免因為終端機/編輯器編碼問題誤植成
# 外觀相似但 code point 不同的字元。
_CJK_RANGES = (
    "一-鿿"   # CJK 統一表意文字
    "㐀-䶿"   # CJK 擴充 A 區
    "豈-﫿"   # CJK 相容表意文字
    "぀-ゟ"   # 平假名
    "゠-ヿ"   # 片假名
    "가-힯"   # 諺文音節
)
_CJK_CHAR = f"[{_CJK_RANGES}]"
_FULLWIDTH_PUNCT = "。，？！：；"

_ASCII_TO_FULLWIDTH = {
    ".": "。",
    ",": "，",
    "?": "？",
    "!": "！",
    ":": "：",
    ";": "；",
}

# 半形標點緊鄰（中間最多容許空白）中日韓文字，且後面沒有立刻接著 ASCII
# 字母／數字（避免誤觸「3.5」「U.S.」「e.g.」網址這類 ASCII 內部的標點）
# 才轉全形；標點與 CJK 字元之間若原本有空白，一併去掉。
# 後面的 lookahead 除了 ASCII 字母/數字之外也要排除其餘半形標點本身，
# 否則「...」「!!」「?!」這種標點連續出現時，只有第一個字元會被轉掉
# （例如「是...」被轉成「是。..」），變成半形/全形混雜。連續標點刻意
# 整串跳過不轉。
_CJK_PUNCT_RE = re.compile(
    f"({_CJK_CHAR})\\s*([{re.escape(''.join(_ASCII_TO_FULLWIDTH))}])"
    r"(?![A-Za-z0-9.,?!:;])")
# 轉換後（或原文就已是全形標點）與下一個 CJK 字元之間如果還留著空白，
# 中日韓文字本來就不加空白分句，一併清掉。
_CJK_GAP_RE = re.compile(
    f"(?<=[{_CJK_RANGES}{_FULLWIDTH_PUNCT}])\\s+(?=[{_CJK_RANGES}])")


def normalize_cjk_punct(text: str) -> str:
    """把緊鄰中日韓文字的半形標點（. , ? ! : ;）轉成全形，並清掉中日韓
    文字之間多餘的空白。

    NLLB/OPUS-MT 逐句翻譯後常常保留來源語言的半形標點與空白，中文/日文/
    韓文字幕卻該用全形——但「3.5」「U.S.」「e.g.」這類 ASCII 內部的標點
    不能被誤轉，所以只在標點緊貼 CJK 字元、且後面不是接著 ASCII 字母或
    數字時才轉換。
    """
    if not text:
        return text

    def _convert(match):
        return match.group(1) + _ASCII_TO_FULLWIDTH[match.group(2)]

    text = _CJK_PUNCT_RE.sub(_convert, text)
    text = _CJK_GAP_RE.sub("", text)
    return text


def postprocess(text: str, tgt: str) -> str:
    """翻譯結果的後處理：目標為中文時轉成台灣繁體；目標為中日韓時，
    緊鄰 CJK 文字的半形標點一律轉全形（且清掉多餘空白）。"""
    if tgt == "zh":
        text = to_traditional(text)
    if tgt in ("zh", "ja", "ko"):
        text = normalize_cjk_punct(text)
    return text


_ELLIPSIS_CHAR = "…"
_SENTENCE_TERMINATORS = ".!?"
_CJK_TERMINATORS = "。！？"


def split_sentences(text: str) -> list:
    """把文字切成一句一句。

    NLLB/OPUS-MT 是句子級模型：多句一次餵進去常常只吐出其中一句
    （實測混進去的長輸入只譯出中間那一句），所以要先切句、逐句翻譯，
    再把結果串回去。

    規則刻意保守（keep it simple）：
    - 「...」（3 個以上的點）或「…」視為單一終止符號：只要後面接空白
      或已到字串結尾就切，不論下一個字的大小寫——刪節號本身就是夠強
      的句界訊號（例：「Wait... what?」要切成「Wait...」「what?」）。
    - 「. ! ?」後面要接空白，且再往後第一個非空白字元是大寫字母、或
      已到字串結尾，才切——避免切在「3.5」這種小數點，或「Mr. smith」
      這種縮寫後接小寫字母的情況。
    - 中日韓句號「。！？」不需要接空白就直接切（CJK 原本就不加空白）。
    """
    if not text:
        return []
    n = len(text)
    pieces = []
    start = 0
    i = 0
    while i < n:
        ch = text[i]
        if ch == _ELLIPSIS_CHAR or (ch == "." and text[i:i + 3] == "..."):
            if ch == _ELLIPSIS_CHAR:
                j = i + 1
            else:
                j = i
                while j < n and text[j] == ".":
                    j += 1
            rest = text[j:]
            if rest == "" or rest[0].isspace():
                piece = text[start:j].strip()
                if piece:
                    pieces.append(piece)
                k = j
                while k < n and text[k].isspace():
                    k += 1
                start = k
                i = k
                continue
            i = j
            continue
        if ch in _CJK_TERMINATORS:
            j = i + 1
            piece = text[start:j].strip()
            if piece:
                pieces.append(piece)
            start = j
            i = j
            continue
        if ch in _SENTENCE_TERMINATORS:
            j = i + 1
            rest = text[j:]
            if rest == "":
                piece = text[start:j].strip()
                if piece:
                    pieces.append(piece)
                start = j
                i = j
                continue
            if rest[0].isspace():
                k = j
                while k < n and text[k].isspace():
                    k += 1
                if k >= n or text[k].isupper():
                    piece = text[start:j].strip()
                    if piece:
                        pieces.append(piece)
                    start = k
                    i = k
                    continue
        i += 1
    if start < n:
        piece = text[start:].strip()
        if piece:
            pieces.append(piece)
    return pieces


def repo_for(engine: str, src: str, tgt: str):
    """回傳 (repo_id, kind)。engine 不存在時丟 KeyError。"""
    spec = ENGINES[engine]
    return spec["repo"].format(src=src, tgt=tgt), spec["kind"]


# 以下三個薄封裝讓 ensure_loaded 的載入流程可以被測試替換，
# 不需要真的連網下載模型或有 GPU。
def _snapshot_download(repo: str) -> str:
    # 程式以 pythonw 執行時沒有 console（sys.stderr 為 None），
    # huggingface_hub 的 tqdm 進度條會在 fp.write 時崩潰——即使模型已快取
    # 也會建立進度條物件，所以一律停用。
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    return snapshot_download(repo)


def _load_tokenizer(path: str, **kwargs):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, **kwargs)


def _make_translator(path: str, device: str, compute_type: str):
    import ctranslate2
    return ctranslate2.Translator(path, device=device, compute_type=compute_type)


class LocalTranslator:
    """CTranslate2 本機翻譯。模型首次使用才下載，之後快取在本機。

    同一個實例可換引擎/語言對，換了會重新載入。所有公開方法皆執行緒安全。
    """

    def __init__(self, engine: str = "nllb-600m",
                 compute_device: str = "auto"):
        self.engine = engine
        self.compute_device = compute_device
        self._lock = threading.RLock()
        self._key = None          # (engine, src, tgt)
        self._translator = None
        self._tokenizer = None
        self._kind = None
        self.device = None        # 實際載入後使用的裝置（"cuda" 或 "cpu"）

    def set_engine(self, engine: str):
        with self._lock:
            if engine != self.engine:
                self.engine = engine
                self._unload()

    def set_compute_device(self, device: str):
        with self._lock:
            if device != self.compute_device:
                self.compute_device = device
                self._unload()

    def is_ready(self, src: str, tgt: str) -> bool:
        return self._key == (self.engine, src, tgt)

    def _unload(self):
        self._key = None
        self._translator = None
        self._tokenizer = None
        self._kind = None
        self.device = None

    @staticmethod
    def _device_candidates(device_pref):
        """回傳 [(device, compute_type)]，依序嘗試。"""
        if device_pref == "cpu":
            return [("cpu", "int8")]
        return [("cuda", "int8_float16"), ("cpu", "int8")]

    @staticmethod
    def _run_batch(translator, tokenizer, kind, texts, tgt) -> list:
        """texts 是一份句子清單，一次呼叫 translate_batch 全部翻完——
        NLLB/OPUS-MT 是句子級模型，逐句分開呼叫沒問題，但多句塞進同一句
        反而常常漏句，所以呼叫端要先用 split_sentences 切開。"""
        sources = [tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
                   for text in texts]
        if kind == "nllb":
            results = translator.translate_batch(
                sources, target_prefix=[[NLLB_CODES[tgt]]] * len(sources),
                beam_size=2)
            hypotheses = [r.hypotheses[0][1:] for r in results]  # 去掉語言標記
        else:
            results = translator.translate_batch(sources, beam_size=2)
            hypotheses = [r.hypotheses[0] for r in results]
        outputs = []
        for hypothesis in hypotheses:
            ids = tokenizer.convert_tokens_to_ids(hypothesis)
            decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
            outputs.append(postprocess(decoded, tgt))
        return outputs

    def ensure_loaded(self, src: str, tgt: str, progress_cb=None):
        """載入（必要時下載）模型。

        下載動輒數百 MB、數分鐘，期間絕不可持鎖：GUI 執行緒會呼叫
        set_engine / set_compute_device，持鎖會讓整個程式「未回應」。
        因此只在「檢查」與「最後掛上」兩個瞬間持鎖，中間全放開；
        下載期間設定被改掉的話，這份成果作廢，下次呼叫會重載。
        """
        with self._lock:
            if self.is_ready(src, tgt):
                return
            engine, device_pref = self.engine, self.compute_device
            repo, kind = repo_for(engine, src, tgt)

        if progress_cb is not None:
            progress_cb(f"正在準備翻譯模型（{repo}）…")
        try:
            path = _snapshot_download(repo)
            register_cuda_dll_dirs()

            tokenizer_kwargs = {}
            if kind == "nllb":
                tokenizer_kwargs["src_lang"] = NLLB_CODES[src]
            tokenizer = _load_tokenizer(path, **tokenizer_kwargs)

            translator = None
            used_device = None
            last_error = None
            for device, compute_type in self._device_candidates(device_pref):
                try:
                    candidate = _make_translator(path, device, compute_type)
                    # CTranslate2 的 CUDA 函式庫是延遲載入：建構成功不代表能用，
                    # 必須真的跑一次推論才會暴露缺 DLL 的錯誤，否則永遠不會退回 CPU
                    self._run_batch(candidate, tokenizer, kind, ["test"], tgt)
                    translator, used_device = candidate, device
                    break
                except Exception as e:
                    last_error = e
            if translator is None:
                raise RuntimeError(str(last_error))
        except Exception as e:
            # 不論是下載失敗（網路）、tokenizer 載入失敗，還是所有裝置都
            # 推論失敗，一律視為致命：以 ModelLoadError（而非訊息字串）
            # 讓呼叫端能明確分辨，不必用字串前綴判斷。
            raise ModelLoadError(f"翻譯模型載入失敗：{e}") from e

        with self._lock:
            if (self.engine, self.compute_device) != (engine, device_pref):
                return  # 設定在載入期間被改了，這份結果作廢，下次呼叫會重載
            self._tokenizer = tokenizer
            self._translator = translator
            self._kind = kind
            self.device = used_device
            self._key = (engine, src, tgt)

    def translate(self, text: str, src: str, tgt: str, progress_cb=None) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        # NLLB/OPUS-MT 是句子級模型：多句一次餵進去常常漏句，所以先切句、
        # 全部句子一次進 _run_batch（一次 translate_batch 呼叫），再串回去
        sentences = split_sentences(text)
        if not sentences:
            return ""
        # ensure_loaded 回來後到取鎖之間，set_engine 可能已經 _unload()，
        # 所以推論前要在鎖內再確認一次；沒就緒就重載（最多兩輪）
        for _ in range(2):
            self.ensure_loaded(src, tgt, progress_cb)
            with self._lock:
                if not self.is_ready(src, tgt):
                    continue
                outputs = self._run_batch(self._translator, self._tokenizer,
                                          self._kind, sentences, tgt)
                sep = "" if tgt in ("zh", "ja", "ko") else " "
                return sep.join(outputs)
        raise RuntimeError("翻譯設定變更過於頻繁，本段跳過")
