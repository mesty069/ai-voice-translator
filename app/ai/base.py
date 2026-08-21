from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranslationResult:
    refined: str
    english: str


class TranslationError(Exception):
    pass


class TranslationProvider(ABC):
    """AI 翻譯供應商的抽象介面：梳理中文並翻譯成英文。"""

    @abstractmethod
    def refine_and_translate(self, text: str) -> TranslationResult:
        """輸入語音辨識出的中文，回傳梳理後中文與英文翻譯。

        失敗時拋出 TranslationError。
        """
        raise NotImplementedError
