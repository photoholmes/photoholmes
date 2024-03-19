from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePreprocessing(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        pass
