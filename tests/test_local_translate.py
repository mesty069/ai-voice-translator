import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import LANGUAGES
from app.core.local_translate import (
    ENGINES,
    ENGINE_LABELS,
    NLLB_CODES,
    LocalTranslator,
    normalize_cjk_punct,
    repo_for,
    split_sentences,
)


def test_nllb_codes_cover_all_app_languages():
    for code, _name in LANGUAGES:
        assert code in NLLB_CODES, f"缺少 {code} 的 NLLB 代碼"


def test_engine_labels_match_engines():
    assert {e for e, _ in ENGINE_LABELS} == set(ENGINES)


def test_repo_for_nllb_is_fixed():
    repo, kind = repo_for("nllb-600m", "en", "zh")
    assert repo == "entai2965/nllb-200-distilled-600M-ctranslate2"
    assert kind == "nllb"


def test_repo_for_opus_is_language_pair_specific():
    repo, kind = repo_for("opus-mt", "en", "zh")
    assert repo == "gaudi/opus-mt-en-zh-ctranslate2"
    assert kind == "opus"


def test_repo_for_unknown_engine_raises():
    with pytest.raises(KeyError):
        repo_for("nope", "en", "zh")


def test_translate_empty_text_returns_empty_without_loading():
    translator = LocalTranslator()
    assert translator.translate("   ", "en", "zh") == ""
    assert not translator.is_ready("en", "zh")


def test_set_engine_invalidates_loaded_model():
    translator = LocalTranslator()
    translator._key = ("nllb-600m", "en", "zh")  # 假裝已載入
    assert translator.is_ready("en", "zh")
    translator.set_engine("opus-mt")
    assert not translator.is_ready("en", "zh")


def test_chinese_uses_simplified_nllb_code():
    # 見 local_translate.py 註解：繁體代碼會系統性截斷，故以簡體解碼再轉繁
    assert NLLB_CODES["zh"] == "zho_Hans"


def test_postprocess_converts_to_taiwan_traditional():
    from app.core.local_translate import postprocess
    assert postprocess("服务器在凌晨3点下降了", "zh") == "伺服器在凌晨3點下降了"


def test_postprocess_leaves_other_languages_untouched():
    from app.core.local_translate import postprocess
    assert postprocess("The server went down.", "en") == "The server went down."


# ---- normalize_cjk_punct（F8：NLLB 逐句翻譯後半形標點沒轉成中文全形） ----

def test_normalize_cjk_punct_smoke_case():
    """實測案例：三句英文翻完後，句號/逗號緊貼中文卻是半形。"""
    text = ("讓我們再來看看.首先,讓我們看看來源中的成分."
            "接下來的關鍵步驟是加熱它.")
    assert normalize_cjk_punct(text) == (
        "讓我們再來看看。首先，讓我們看看來源中的成分。"
        "接下來的關鍵步驟是加熱它。")


def test_normalize_cjk_punct_keeps_decimal_point():
    assert normalize_cjk_punct("版本 3.5 已發布.") == "版本 3.5 已發布。"


def test_normalize_cjk_punct_keeps_abbreviation_intact():
    result = normalize_cjk_punct("見 e.g. 這個.")
    assert "e.g." in result
    assert result == "見 e.g. 這個。"


def test_normalize_cjk_punct_comma_and_strips_gap():
    assert normalize_cjk_punct("你好, 世界") == "你好，世界"


def test_normalize_cjk_punct_is_idempotent():
    assert normalize_cjk_punct("你好。世界") == "你好。世界"


def test_normalize_cjk_punct_english_target_unchanged_via_postprocess():
    from app.core.local_translate import postprocess
    text = "Hello, world. Nice to meet you."
    assert postprocess(text, "en") == text


def test_falls_back_to_cpu_when_cuda_fails_at_inference(monkeypatch, tmp_path):
    """CTranslate2 的 CUDA 錯誤是延遲發生的：建構成功、推論才失敗，
    所以只包住建構子的 try/except 永遠不會退回 CPU。"""
    from app.core import local_translate as lt

    class FakeTokenizer:
        def encode(self, text):
            return [1, 2]

        def convert_ids_to_tokens(self, ids):
            return ["a", "b"]

        def convert_tokens_to_ids(self, tokens):
            return [1]

        def decode(self, ids, skip_special_tokens=True):
            return "翻譯結果"

    class FakeResult:
        hypotheses = [["lang", "x"]]

    class FakeTranslator:
        def __init__(self, device):
            self.device = device

        def translate_batch(self, sources, **kwargs):
            if self.device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found")
            return [FakeResult()]

    monkeypatch.setattr(lt, "_snapshot_download", lambda repo: str(tmp_path))
    monkeypatch.setattr(lt, "_load_tokenizer",
                        lambda path, **kwargs: FakeTokenizer())
    monkeypatch.setattr(lt, "_make_translator",
                        lambda path, device, compute_type: FakeTranslator(device))

    translator = lt.LocalTranslator()
    assert translator.translate("hello", "en", "zh") == "翻譯結果"
    assert translator.device == "cpu"


def test_snapshot_download_survives_no_console(monkeypatch, tmp_path):
    """pythonw 無 console 時 sys.stderr 為 None，進度條不得讓下載崩潰。"""
    import sys
    from app.core import local_translate as lt

    class FakeHub:
        def __init__(self):
            self.progress_disabled = False

        def snapshot_download(self, repo):
            # 模擬 huggingface_hub：若進度條沒被停用就建立 tqdm 寫 stderr
            if not self.progress_disabled:
                sys.stderr.write("progress")  # sys.stderr 是 None → AttributeError
            return str(tmp_path)

        def disable_progress_bars(self):
            self.progress_disabled = True

    hub = FakeHub()
    import huggingface_hub, huggingface_hub.utils
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub.snapshot_download)
    monkeypatch.setattr(huggingface_hub.utils, "disable_progress_bars",
                        hub.disable_progress_bars)
    monkeypatch.setattr(sys, "stderr", None)
    assert lt._snapshot_download("some/repo") == str(tmp_path)


def test_ensure_loaded_does_not_hold_lock_during_download(monkeypatch, tmp_path):
    """下載模型（可能好幾分鐘）期間不得持鎖，否則 GUI 執行緒呼叫
    set_engine / set_compute_device 會整個卡住（未回應）。"""
    from app.core import local_translate as lt

    class FakeTokenizer:
        def encode(self, text):
            return [1]

        def convert_ids_to_tokens(self, ids):
            return ["a"]

        def convert_tokens_to_ids(self, tokens):
            return [1]

        def decode(self, ids, skip_special_tokens=True):
            return "ok"

    class FakeResult:
        hypotheses = [["lang", "x"]]

    class FakeTranslator:
        def translate_batch(self, sources, **kwargs):
            return [FakeResult()]

    translator = lt.LocalTranslator()
    observed = {}

    def fake_download(repo):
        # 模擬下載中：此時「別的執行緒」必須也能取得鎖。
        # _lock 是 RLock，同一條執行緒試取一定成功，所以必須跨執行緒檢查。
        def probe():
            observed["free"] = translator._lock.acquire(blocking=False)
            if observed["free"]:
                translator._lock.release()

        t = threading.Thread(target=probe)
        t.start()
        t.join(timeout=2.0)
        return str(tmp_path)

    monkeypatch.setattr(lt, "_snapshot_download", fake_download)
    monkeypatch.setattr(lt, "_load_tokenizer",
                        lambda path, **kwargs: FakeTokenizer())
    monkeypatch.setattr(lt, "_make_translator",
                        lambda path, device, compute_type: FakeTranslator())

    translator.ensure_loaded("en", "zh")
    assert observed["free"] is True, "下載期間仍持有鎖，GUI 會被凍結"


def test_translate_reloads_when_model_is_unloaded_mid_call(monkeypatch, tmp_path):
    """ensure_loaded 回來後、推論前，若別的執行緒 set_engine 清掉模型，
    不能拿 None 去推論（舊版會 AttributeError），要重載後再跑。"""
    from app.core import local_translate as lt

    class FakeTokenizer:
        def encode(self, text):
            return [1]

        def convert_ids_to_tokens(self, ids):
            return ["a"]

        def convert_tokens_to_ids(self, tokens):
            return [1]

        def decode(self, ids, skip_special_tokens=True):
            return "翻譯結果"

    class FakeResult:
        hypotheses = [["lang", "x"]]

    class FakeTranslator:
        def translate_batch(self, sources, **kwargs):
            return [FakeResult()]

    downloads = []
    monkeypatch.setattr(lt, "_snapshot_download",
                        lambda repo: downloads.append(repo) or str(tmp_path))
    monkeypatch.setattr(lt, "_load_tokenizer",
                        lambda path, **kwargs: FakeTokenizer())
    monkeypatch.setattr(lt, "_make_translator",
                        lambda path, device, compute_type: FakeTranslator())

    translator = lt.LocalTranslator(compute_device="cpu")
    original_is_ready = translator.is_ready
    tripped = []

    def racy_is_ready(src, tgt):
        ready = original_is_ready(src, tgt)
        if ready and not tripped:
            tripped.append(True)
            translator._unload()   # 模擬此刻另一條執行緒 set_engine
            return False
        return ready

    translator.is_ready = racy_is_ready
    assert translator.translate("hello", "en", "zh") == "翻譯結果"
    assert len(downloads) == 2, "模型被清掉後應重新載入一次"


# ---- split_sentences（F7：NLLB/OPUS-MT 是句子級模型，多句常常漏句） ----

def test_split_sentences_multi_sentence_english():
    text = ("Let's review it again. First, let's look at the ingredients "
            "in the source. The next key step is...")
    assert split_sentences(text) == [
        "Let's review it again.",
        "First, let's look at the ingredients in the source.",
        "The next key step is...",
    ]


def test_split_sentences_does_not_split_decimal_point():
    assert split_sentences("Version 3.5 is out. Great.") == [
        "Version 3.5 is out.",
        "Great.",
    ]


def test_split_sentences_ellipsis_is_single_terminator():
    assert split_sentences("Wait... what?") == ["Wait...", "what?"]


def test_split_sentences_cjk_terminators():
    assert split_sentences("你好。再見！") == ["你好。", "再見！"]


def test_split_sentences_single_sentence():
    assert split_sentences("Hello there") == ["Hello there"]
    assert split_sentences("Hello there.") == ["Hello there."]


def test_split_sentences_empty_input():
    assert split_sentences("") == []
    assert split_sentences(None) == []


def test_translate_batches_all_sentences_in_one_call(monkeypatch, tmp_path):
    """多句輸入要一次呼叫 translate_batch 把所有句子一起送進去，
    不能逐句各自呼叫（那樣就失去意義，而且更慢）。target=zh 時句子間
    直接串接，不加空白。"""
    from app.core import local_translate as lt

    class FakeTokenizer:
        def encode(self, text):
            return [len(text)]

        def convert_ids_to_tokens(self, ids):
            return [str(i) for i in ids]

        def convert_tokens_to_ids(self, tokens):
            return [int(t) for t in tokens]

        def decode(self, ids, skip_special_tokens=True):
            return f"譯[{ids[0]}]"

    class FakeResult:
        def __init__(self, tag):
            self.hypotheses = [["lang", tag]]

    captured = {}

    class FakeTranslator:
        def translate_batch(self, sources, **kwargs):
            captured["sources"] = sources
            captured["kwargs"] = kwargs
            return [FakeResult(src[-1]) for src in sources]

    monkeypatch.setattr(lt, "_snapshot_download", lambda repo: str(tmp_path))
    monkeypatch.setattr(lt, "_load_tokenizer",
                        lambda path, **kwargs: FakeTokenizer())
    monkeypatch.setattr(lt, "_make_translator",
                        lambda path, device, compute_type: FakeTranslator())

    translator = lt.LocalTranslator(compute_device="cpu")
    text = "One. Two. Three."  # 三句：len 各為 4, 4, 6
    result = translator.translate(text, "en", "zh")

    assert len(captured["sources"]) == 3, "三句要在同一次 translate_batch 送出"
    assert captured["kwargs"]["target_prefix"] == [["zho_Hans"]] * 3
    assert result == "譯[4]譯[4]譯[6]"


def test_translate_joins_non_cjk_target_with_space(monkeypatch, tmp_path):
    """目標語言不是中日韓時，句子之間要用空白接回去。"""
    from app.core import local_translate as lt

    class FakeTokenizer:
        def encode(self, text):
            return [len(text)]

        def convert_ids_to_tokens(self, ids):
            return [str(i) for i in ids]

        def convert_tokens_to_ids(self, tokens):
            return [int(t) for t in tokens]

        def decode(self, ids, skip_special_tokens=True):
            return f"tr{ids[0]}"

    class FakeResult:
        def __init__(self, tag):
            self.hypotheses = [tag]  # opus：不去掉語言標記那個 slice

    captured = {}

    class FakeTranslator:
        def translate_batch(self, sources, **kwargs):
            captured["sources"] = sources
            return [FakeResult(src) for src in sources]

    monkeypatch.setattr(lt, "_snapshot_download", lambda repo: str(tmp_path))
    monkeypatch.setattr(lt, "_load_tokenizer",
                        lambda path, **kwargs: FakeTokenizer())
    monkeypatch.setattr(lt, "_make_translator",
                        lambda path, device, compute_type: FakeTranslator())

    translator = lt.LocalTranslator(engine="opus-mt", compute_device="cpu")
    result = translator.translate("Hi there. Bye now.", "en", "fr")

    # "Hi there." 長度 9、"Bye now." 長度 8 → FakeTokenizer 以長度當 id
    assert len(captured["sources"]) == 2
    assert result == "tr9 tr8"
