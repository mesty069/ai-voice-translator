import json

import requests

from .base import (
    GrammarResult,
    TranslationError,
    TranslationProvider,
    TranslationResult,
)

def translate_prompt(source_name: str, target_name: str) -> str:
    return (
        f"你是一個翻譯助手。使用者給你的{source_name}來自語音辨識，"
        "可能含有同音錯字、缺標點或斷句錯誤。"
        f"請先把它梳理成通順正確的{source_name}：修正錯字與斷句，"
        "並依語氣補上完整正確的標點符號（逗號、句號、問號、驚嘆號等），"
        "保持原意，不要增減內容。"
        f"再把梳理後的內容翻譯成自然流暢的{target_name}，"
        "翻譯同樣要有正確完整的標點符號。\n"
        '只回傳 JSON，格式：{"refined": "梳理後的原文", '
        '"translation": "翻譯結果"}'
    )


def grammar_prompt(language_name: str) -> str:
    return (
        f"你是{language_name}文法檢查助手。使用者給你的{language_name}"
        "來自語音辨識。判斷句子是否有文法錯誤（時態、單複數、用詞明顯"
        "錯誤等都算）。口語化省略、大小寫與標點差異不算錯誤。"
        "有錯誤時給出修正後的完整句子，保持原本的語氣與意思，"
        "不要改寫成別的說法。\n"
        '只回傳 JSON，格式：{"has_errors": true 或 false, '
        '"corrected": "修正後的完整句子（沒錯誤時原句照抄）"}'
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

    def _chat_json(self, system_prompt: str, text: str) -> dict:
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
                        {"role": "system", "content": system_prompt},
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

        payload = resp.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise TranslationError(f"DeepSeek 回應格式異常：{payload}") from e
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise TranslationError(f"無法解析 AI 回傳的 JSON：{content[:200]}") from e

    def refine_and_translate(self, text: str, source_name: str = "繁體中文",
                             target_name: str = "英文") -> TranslationResult:
        data = self._chat_json(translate_prompt(source_name, target_name), text)
        try:
            refined = str(data["refined"]).strip()
            translation = str(data["translation"]).strip()
        except (KeyError, TypeError) as e:
            raise TranslationError(f"AI 回傳缺少欄位：{data}") from e
        if not translation:
            raise TranslationError("AI 回傳的翻譯是空的。")
        return TranslationResult(refined=refined, english=translation)

    def check_grammar(self, text: str,
                      language_name: str = "英文") -> GrammarResult:
        data = self._chat_json(grammar_prompt(language_name), text)
        try:
            has_errors = bool(data["has_errors"])
            corrected = str(data["corrected"]).strip()
        except (KeyError, TypeError) as e:
            raise TranslationError(f"AI 回傳缺少欄位：{data}") from e
        if has_errors and not corrected:
            raise TranslationError("AI 回報有錯誤但沒有給修正句。")
        return GrammarResult(has_errors=has_errors, corrected=corrected)
