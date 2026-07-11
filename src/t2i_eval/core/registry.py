from .evaluator import BaseEvaluator
from .model import BaseModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {}
EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {}


def register_model(name: str):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def register_evaluator(name: str):
    def decorator(cls):
        EVALUATOR_REGISTRY[name] = cls
        cls._registry_name = name
        return cls

    return decorator


def get_model_class(name: str) -> type[BaseModel] | None:
    return MODEL_REGISTRY.get(name)


def get_evaluator_class(name: str) -> type[BaseEvaluator] | None:
    return EVALUATOR_REGISTRY.get(name)
