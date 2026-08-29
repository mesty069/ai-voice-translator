import threading

# 程式的語言代碼 → NLLB-200 的語言代碼
NLLB_CODES = {
    "zh": "zho_Hant",
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


def repo_for(engine: str, src: str, tgt: str):
    """回傳 (repo_id, kind)。engine 不存在時丟 KeyError。"""
    spec = ENGINES[engine]
    return spec["repo"].format(src=src, tgt=tgt), spec["kind"]


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

    def _device_candidates(self):
        """回傳 [(device, compute_type)]，依序嘗試。"""
        if self.compute_device == "cpu":
            return [("cpu", "int8")]
        return [("cuda", "int8_float16"), ("cpu", "int8")]

    def ensure_loaded(self, src: str, tgt: str, progress_cb=None):
        with self._lock:
            if self.is_ready(src, tgt):
                return
            repo, kind = repo_for(self.engine, src, tgt)
            if progress_cb is not None:
                progress_cb(f"正在準備翻譯模型（{repo}）…")
            from huggingface_hub import snapshot_download
            path = snapshot_download(repo)

            import ctranslate2
            from transformers import AutoTokenizer

            last_error = None
            translator = None
            for device, compute_type in self._device_candidates():
                try:
                    translator = ctranslate2.Translator(
                        path, device=device, compute_type=compute_type)
                    break
                except Exception as e:  # GPU 不可用或記憶體不足 → 退回 CPU
                    last_error = e
            if translator is None:
                raise RuntimeError(f"翻譯模型載入失敗：{last_error}")

            tokenizer_kwargs = {}
            if kind == "nllb":
                tokenizer_kwargs["src_lang"] = NLLB_CODES[src]
            self._tokenizer = AutoTokenizer.from_pretrained(
                path, **tokenizer_kwargs)
            self._translator = translator
            self._kind = kind
            self._key = (self.engine, src, tgt)

    def translate(self, text: str, src: str, tgt: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        self.ensure_loaded(src, tgt)
        with self._lock:
            tokenizer = self._tokenizer
            source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
            if self._kind == "nllb":
                results = self._translator.translate_batch(
                    [source], target_prefix=[[NLLB_CODES[tgt]]], beam_size=2)
                hypothesis = results[0].hypotheses[0][1:]  # 去掉語言標記
            else:
                results = self._translator.translate_batch(
                    [source], beam_size=2)
                hypothesis = results[0].hypotheses[0]
            ids = tokenizer.convert_tokens_to_ids(hypothesis)
            return tokenizer.decode(ids, skip_special_tokens=True).strip()
