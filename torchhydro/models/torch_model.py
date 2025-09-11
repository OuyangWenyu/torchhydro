import numpy as np
from typing import Dict, Any, Optional
from torchhydro.models.registry import register_model
from torchhydro.models.simple_lstm import SimpleLSTM

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is not installed. Please install it to use PyTorch models.")


@register_model("pytorch")
class PytorchModel:
    """
    A wrapper for PyTorch models to be used in the UnifiedSimulator.
    """

    def __init__(self, model_config: Dict[str, Any], basin_config: Any = None):
        """
        Initialize the PyTorch model wrapper.

        Args:
            model_config (Dict[str, Any]): The model configuration.
                It must contain a 'model_path' key with the path to the saved PyTorch model.
            basin_config (Any, optional): The basin configuration. Not used by this model.
        """
        self.model_config = model_config

        self.model_path = self.model_config.get('model_path')
        if not self.model_path:
            raise ValueError("'model_path' not found in model_config for PytorchModel")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(self.model_path)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load a PyTorch model from a file.

        Args:
            model_path (str): The path to the saved model file.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """
        try:
            # Load the model. The file should contain the model state_dict or the whole model.
            # For simplicity, we assume the whole model is saved.
            model_params = self.model_config.get("model_params", {})
            input_size = model_params.get("input_size")
            hidden_size = model_params.get("hidden_size")
            output_size = model_params.get("out_size")
            dropout = model_params.get("dropout", 0.0)

            if input_size is None or hidden_size is None or output_size is None:
                raise ValueError(
                    "'model_params' must contain 'input_size', 'hidden_size', 'output_size'"
                )

            model = SimpleLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                dr=dropout
            ).to(self.device)

            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()

            return model
        except Exception as e:
            raise IOError(f"Error loading PyTorch model from {model_path}: {e}")

    def simulate(
        self,
        inputs: np.ndarray,
        qobs: Optional[np.ndarray] = None,
        warmup_length: int = 0,
        is_event_data: bool = False,
        return_intermediate: bool = False,
        return_warmup_states: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the PyTorch model simulation.

        Args:
            inputs (np.ndarray): Input data array with shape [time, basin, features].
            qobs (Optional[np.ndarray]): Observed data. Not used by the model but kept for interface compatibility.
            warmup_length (int): The length of the warmup period. The model will run on the whole sequence.
            is_event_data (bool): Flag indicating if the data is event-based. Not used.
            return_intermediate (bool): Flag to return intermediate results. Not used.
            return_warmup_states (bool): Flag to return warmup states. Not used.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary containing the simulation results.
        """
        with torch.no_grad():
            # Convert numpy array to torch tensor
            input_tensor = torch.from_numpy(inputs).float().to(self.device)

            # Assume the model returns a tensor of shape [time, basin, 1]
            qsim_tensor = self.model(input_tensor)

            # Convert the output tensor back to a numpy array
            qsim = qsim_tensor.cpu().numpy()

        return {
            "qsim": qsim,
            "qobs": qobs,
        }
