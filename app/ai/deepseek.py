import json

import requests

from .base import TranslationProvider, TranslationResult, TranslationError

SYSTEM_PROMPT = (
    "你是一個中翻英助手。使用者給你的中文來自語音辨識，可能含有同音錯字、"
    "缺標點或斷句錯誤。請先把它梳理成通順正確的繁體中文（保持原意，不要增減內容），"
    "再把梳理後的中文翻譯成自然流暢的英文。\n"
    '只回傳 JSON，格式：{"refined": "梳理後的繁體中文", "english": "英文翻譯"}'
)


class DeepSeekProvider(TranslationProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", timeout: float = 60.0):
        if not api_key:
            raise TranslationError("尚未設定 DeepSeek API key，請到設定頁填入。")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def refine_and_translate(self, text: str) -> TranslationResult:
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.3,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise TranslationError(f"無法連線到 DeepSeek API：{e}") from e

        if resp.status_code != 200:
            raise TranslationError(
                f"DeepSeek API 回應錯誤（HTTP {resp.status_code}）：{resp.text[:200]}")

        return self._parse_response(resp.json())

    @staticmethod
    def _parse_response(payload: dict) -> TranslationResult:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise TranslationError(f"DeepSeek 回應格式異常：{payload}") from e
        try:
            data = json.loads(content)
            refined = str(data["refined"]).strip()
            english = str(data["english"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise TranslationError(f"無法解析 AI 回傳的 JSON：{content[:200]}") from e
        if not english:
            raise TranslationError("AI 回傳的英文翻譯是空的。")
        return TranslationResult(refined=refined, english=english)
