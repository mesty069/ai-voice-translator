import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.base import TranslationError
from app.ai.deepseek import DeepSeekProvider
from app.ai.factory import create_provider


def _payload(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


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
    content = json.dumps({"refined": "你好嗎？", "english": "How are you?"},
                         ensure_ascii=False)
    result = DeepSeekProvider._parse_response(_payload(content))
    assert result.refined == "你好嗎？"
    assert result.english == "How are you?"


def test_parse_invalid_json_raises():
    with pytest.raises(TranslationError):
        DeepSeekProvider._parse_response(_payload("not json"))


def test_parse_missing_field_raises():
    with pytest.raises(TranslationError):
        DeepSeekProvider._parse_response(_payload(json.dumps({"refined": "x"})))


def test_parse_empty_english_raises():
    with pytest.raises(TranslationError):
        DeepSeekProvider._parse_response(
            _payload(json.dumps({"refined": "x", "english": ""})))
