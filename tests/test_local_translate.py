import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import LANGUAGES
from app.core.local_translate import (
    ENGINES,
    ENGINE_LABELS,
    NLLB_CODES,
    LocalTranslator,
    repo_for,
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
