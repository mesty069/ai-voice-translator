from .base import TranslationProvider, TranslationError
from .deepseek import DeepSeekProvider


def create_provider(ai_config: dict) -> TranslationProvider:
    """依 config 的 ai 區段建立對應的 TranslationProvider。

    ai_config 範例：
    {"provider": "deepseek", "deepseek": {"api_key": "...", "base_url": "...", "model": "..."}}
    """
    provider_name = (ai_config.get("provider") or "").lower()
    if provider_name == "deepseek":
        cfg = ai_config.get("deepseek", {})
        return DeepSeekProvider(
            api_key=cfg.get("api_key", ""),
            base_url=cfg.get("base_url", "https://api.deepseek.com"),
            model=cfg.get("model", "deepseek-chat"),
        )
    raise TranslationError(f"不支援的 AI provider：{provider_name!r}")
