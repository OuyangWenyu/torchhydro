"""
Author: Wenyu Ouyang
Date: 2025-01-13
Description: Lightning Fabric wrapper for debugging and distributed training
FilePath: /torchhydro/torchhydro/trainers/fabric_wrapper.py
"""

import os
import torch
from typing import Any, Dict, Optional, Union


class FabricWrapper:
    """
    A wrapper class that can switch between Lightning Fabric and normal PyTorch operations
    based on configuration settings.
    """

    def __init__(self, use_fabric: bool = True, fabric_config: Optional[Dict] = None):
        """
        Initialize the Fabric wrapper.

        Parameters
        ----------
        use_fabric : bool
            Whether to use Lightning Fabric or normal PyTorch operations
        fabric_config : Optional[Dict]
            Configuration for Fabric (devices, strategy, etc.)
        """
        self.use_fabric = use_fabric
        self.fabric_config = fabric_config or {}
        self._fabric = None
        self._device = None

        if self.use_fabric:
            self._init_fabric()
        else:
            self._init_pytorch()

    def _init_fabric(self):
        """Initialize Lightning Fabric"""
        try:
            import lightning as L

            # Default fabric configuration
            default_config = {
                "accelerator": "auto",
                "devices": "auto",
                "strategy": "auto",
                "precision": "32-true",
            }

            # Update with user config
            default_config.update(self.fabric_config)

            self._fabric = L.Fabric(**default_config)
            print("✅ Lightning Fabric initialized successfully")

        except ImportError:
            print("❌ Lightning not found, falling back to normal PyTorch")
            self.use_fabric = False
            self._init_pytorch()

    def _init_pytorch(self):
        """Initialize normal PyTorch setup"""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Normal PyTorch initialized, using device: {self._device}")

    def setup_module(self, model):
        """Setup model for training"""
        if self.use_fabric:
            return self._fabric.setup_module(model)
        else:
            return model.to(self._device)

    def setup_optimizers(self, optimizer):
        """Setup optimizer"""
        if self.use_fabric:
            return self._fabric.setup_optimizers(optimizer)
        else:
            return optimizer

    def setup_dataloaders(self, *dataloaders):
        """Setup dataloaders"""
        if self.use_fabric:
            return self._fabric.setup_dataloaders(*dataloaders)
        else:
            return dataloaders

    def save(self, path: str, state_dict: Dict[str, Any]):
        """Save model state"""
        if self.use_fabric:
            self._fabric.save(path, state_dict)
        else:
            torch.save(state_dict, path)

    def load(self, path: str, model: Optional[torch.nn.Module] = None):
        """Load model state"""
        if self.use_fabric:
            return self._fabric.load(path, model)
        else:
            return torch.load(path, map_location=self._device)

    def load_raw(self, path: str, model: torch.nn.Module):
        """Load raw model weights"""
        if self.use_fabric:
            checkpoint = self._fabric.load(path)
            model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load(path, map_location=self._device)
            model.load_state_dict(checkpoint)

    def launch(self, fn=None, *args, **kwargs):
        """Launch training function"""
        if self.use_fabric:
            if fn is None:
                # This is called without a function, just launch fabric
                return self._fabric.launch()
            else:
                return self._fabric.launch(fn, *args, **kwargs)
        else:
            # Normal PyTorch, just call the function directly
            if fn is not None:
                return fn(*args, **kwargs)
            else:
                return None

    def backward(self, loss):
        """Backward pass"""
        if self.use_fabric:
            self._fabric.backward(loss)
        else:
            loss.backward()

    def clip_gradients(self, model, optimizer, max_norm: float = 1.0):
        """Clip gradients"""
        if self.use_fabric:
            self._fabric.clip_gradients(model, optimizer, max_norm=max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    @property
    def device(self):
        """Get current device"""
        if self.use_fabric:
            return self._fabric.device
        else:
            return self._device

    @property
    def local_rank(self):
        """Get local rank"""
        if self.use_fabric:
            return self._fabric.local_rank
        else:
            return 0

    @property
    def global_rank(self):
        """Get global rank"""
        if self.use_fabric:
            return self._fabric.global_rank
        else:
            return 0

    @property
    def world_size(self):
        """Get world size"""
        if self.use_fabric:
            return self._fabric.world_size
        else:
            return 1

    def barrier(self):
        """Synchronization barrier"""
        if self.use_fabric:
            self._fabric.barrier()
        else:
            pass  # No barrier needed for single process

    def print(self, *args, **kwargs):
        """Print only on rank 0"""
        if self.use_fabric:
            self._fabric.print(*args, **kwargs)
        else:
            print(*args, **kwargs)


def create_fabric_wrapper(training_cfgs: Dict) -> FabricWrapper:
    """
    Create a fabric wrapper based on training configuration.

    Parameters
    ----------
    training_cfgs : Dict
        Training configuration dictionary

    Returns
    -------
    FabricWrapper
        Initialized fabric wrapper
    """
    # Check if we should use fabric
    use_fabric = training_cfgs.get("use_fabric", True)

    # For debugging, disable fabric
    if training_cfgs.get("debug_mode", False):
        use_fabric = False
        print("🐛 Debug mode enabled - disabling Lightning Fabric")

    # Check if we have multiple devices
    devices = training_cfgs.get("device", [0])
    if isinstance(devices, list) and len(devices) == 1:
        # Single device, we might not need fabric
        if not training_cfgs.get("force_fabric", False):
            print("📱 Single device detected - you can disable Fabric for debugging")

    # Fabric configuration
    fabric_config = {
        "devices": devices if isinstance(devices, list) else [devices],
        "strategy": training_cfgs.get("strategy", "auto"),
        "precision": training_cfgs.get("precision", "32-true"),
        "accelerator": training_cfgs.get("accelerator", "auto"),
    }

    return FabricWrapper(use_fabric=use_fabric, fabric_config=fabric_config)
