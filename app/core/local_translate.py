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


_opencc = None


def to_traditional(text: str) -> str:
    """簡體 → 台灣繁體（含用語轉換，如 服务器→伺服器）。"""
    global _opencc
    if _opencc is None:
        from opencc import OpenCC
        _opencc = OpenCC("s2twp")
    return _opencc.convert(text)


def postprocess(text: str, tgt: str) -> str:
    """翻譯結果的後處理：目標為中文時轉成台灣繁體。"""
    if tgt == "zh":
        return to_traditional(text)
    return text


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

    def _device_candidates(self):
        """回傳 [(device, compute_type)]，依序嘗試。"""
        if self.compute_device == "cpu":
            return [("cpu", "int8")]
        return [("cuda", "int8_float16"), ("cpu", "int8")]

    @staticmethod
    def _run_batch(translator, tokenizer, kind, text, tgt) -> str:
        source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        if kind == "nllb":
            results = translator.translate_batch(
                [source], target_prefix=[[NLLB_CODES[tgt]]], beam_size=2)
            hypothesis = results[0].hypotheses[0][1:]  # 去掉語言標記
        else:
            results = translator.translate_batch([source], beam_size=2)
            hypothesis = results[0].hypotheses[0]
        ids = tokenizer.convert_tokens_to_ids(hypothesis)
        decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
        return postprocess(decoded, tgt)

    def ensure_loaded(self, src: str, tgt: str, progress_cb=None):
        with self._lock:
            if self.is_ready(src, tgt):
                return
            repo, kind = repo_for(self.engine, src, tgt)
            if progress_cb is not None:
                progress_cb(f"正在準備翻譯模型（{repo}）…")
            path = _snapshot_download(repo)
            register_cuda_dll_dirs()

            tokenizer_kwargs = {}
            if kind == "nllb":
                tokenizer_kwargs["src_lang"] = NLLB_CODES[src]
            tokenizer = _load_tokenizer(path, **tokenizer_kwargs)

            translator = None
            used_device = None
            last_error = None
            for device, compute_type in self._device_candidates():
                try:
                    candidate = _make_translator(path, device, compute_type)
                    # CTranslate2 的 CUDA 函式庫是延遲載入：建構成功不代表能用，
                    # 必須真的跑一次推論才會暴露缺 DLL 的錯誤，否則永遠不會退回 CPU
                    self._run_batch(candidate, tokenizer, kind, "test", tgt)
                    translator, used_device = candidate, device
                    break
                except Exception as e:
                    last_error = e
            if translator is None:
                raise RuntimeError(f"翻譯模型載入失敗：{last_error}")

            self._tokenizer = tokenizer
            self._translator = translator
            self._kind = kind
            self.device = used_device
            self._key = (self.engine, src, tgt)

    def translate(self, text: str, src: str, tgt: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        self.ensure_loaded(src, tgt)
        with self._lock:
            return self._run_batch(self._translator, self._tokenizer,
                                   self._kind, text, tgt)
