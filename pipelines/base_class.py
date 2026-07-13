from abc import ABC, abstractmethod
from typing import List

from data_models import EvaluationEntry


class Pipeline(ABC):
    @abstractmethod
    async def run_pipeline(
        self, dataset: List[EvaluationEntry]
    ) -> List[EvaluationEntry]:
        pass
