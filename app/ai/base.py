from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranslationResult:
    refined: str
    english: str


@dataclass
class GrammarResult:
    has_errors: bool
    corrected: str


class TranslationError(Exception):
    pass


class TranslationProvider(ABC):
    """AI 翻譯供應商的抽象介面：梳理中文並翻譯成英文。"""

    @abstractmethod
    def refine_and_translate(self, text: str, source_name: str = "繁體中文",
                             target_name: str = "英文") -> TranslationResult:
        """輸入語音辨識出的母語文字，梳理後翻譯成目標語言。

        失敗時拋出 TranslationError。
        """
        raise NotImplementedError

    def check_grammar(self, text: str,
                      language_name: str = "英文") -> GrammarResult:
        """檢查指定語言句子的文法，回傳是否有錯與修正後的句子。

        失敗時拋出 TranslationError。
        """
        raise NotImplementedError
