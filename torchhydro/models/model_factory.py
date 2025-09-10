from typing import Dict, Any, Type
from torchhydro.models.traditional_model import TraditionalModel
from torchhydro.models.registry import MODEL_REGISTRY
# We need to handle the case where torch is not installed
try:
    from torch_model import PytorchModel

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def register_model(name: str):
    """装饰器：把模型类注册到 MODEL_REGISTRY"""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def model_factory(model_config: Dict[str, Any], basin_config: Any = None) -> Any:
    """
    Factory function to instantiate a model based on its type.

    Args:
        model_config (Dict[str, Any]): The model configuration.
            It must contain a 'type' key, e.g., 'traditional' or 'pytorch'.
        basin_config (Any, optional): The basin configuration.

    Returns:
        An instance of a model wrapper (e.g., TraditionalModel, PytorchModel).
    """
    model_type = model_config.get("type", "traditional") # Default to traditional for backward compatibility

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Available models: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_type]
    return model_class(model_config, basin_config)
