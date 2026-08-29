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
