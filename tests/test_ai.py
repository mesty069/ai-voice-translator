import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.base import TranslationError
from app.ai.deepseek import DeepSeekProvider
from app.ai.factory import create_provider


def _provider_with_response(data: dict) -> DeepSeekProvider:
    provider = DeepSeekProvider(api_key="sk-x")
    provider._chat_json = lambda system_prompt, text: data
    return provider


def test_factory_creates_deepseek():
    provider = create_provider({
        "provider": "deepseek",
        "deepseek": {"api_key": "sk-x", "base_url": "https://api.deepseek.com",
                     "model": "deepseek-chat"},
    })
    assert isinstance(provider, DeepSeekProvider)
    assert provider.model == "deepseek-chat"


def test_factory_rejects_unknown_provider():
    with pytest.raises(TranslationError):
        create_provider({"provider": "nope"})


def test_deepseek_requires_api_key():
    with pytest.raises(TranslationError):
        DeepSeekProvider(api_key="")


def test_parse_valid_response():
    provider = _provider_with_response(
        {"refined": "你好嗎？", "translation": "How are you?"})
    result = provider.refine_and_translate("x", "繁體中文", "英文")
    assert result.refined == "你好嗎？"
    assert result.english == "How are you?"


def test_parse_missing_field_raises():
    with pytest.raises(TranslationError):
        _provider_with_response({"refined": "x"}).refine_and_translate("x")


def test_parse_empty_translation_raises():
    with pytest.raises(TranslationError):
        _provider_with_response(
            {"refined": "x", "translation": ""}).refine_and_translate("x")


def test_prompts_use_language_names():
    from app.ai.deepseek import grammar_prompt, translate_prompt
    p = translate_prompt("繁體中文", "日文")
    assert "繁體中文" in p and "日文" in p
    g = grammar_prompt("日文")
    assert "日文" in g


def test_grammar_no_errors():
    result = _provider_with_response(
        {"has_errors": False, "corrected": "Same."}).check_grammar("Same.")
    assert not result.has_errors


def test_grammar_with_errors():
    result = _provider_with_response(
        {"has_errors": True, "corrected": "He went home."}
    ).check_grammar("He go home.")
    assert result.has_errors and result.corrected == "He went home."


def test_grammar_missing_field_raises():
    with pytest.raises(TranslationError):
        _provider_with_response({"has_errors": True}).check_grammar("x")
