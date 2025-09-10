from typing import Dict, Any, Callable

MODEL_REGISTRY: Dict[str, Callable] = {}

def register_model(name: str):
    """装饰器：注册模型"""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator
