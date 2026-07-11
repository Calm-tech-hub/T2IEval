from abc import ABC, abstractmethod
from typing import Any

from .model import BaseModel
from .schema import EvaluatorConfig


class BaseEvaluator(ABC):
    """
    Base class for all evaluators.
    """

    def __init__(self, config: EvaluatorConfig):
        pass

    @abstractmethod
    def evaluate(self, model: BaseModel) -> dict[str, Any] | None:
        """
        Evaluate the model and return the metrics.

        Args:
            model (BaseModel): The model to evaluate.

        Returns:
            Dict[str, Any] | None: A dictionary containing evaluation metrics, or None if not rank 0.
        """
        pass
