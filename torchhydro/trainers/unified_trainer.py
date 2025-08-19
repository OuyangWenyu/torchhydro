from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from torchhydro.models.model_dict_function import pytorch_model_dict
from torchhydro.trainers.train_utils import get_preds_to_be_eval, model_infer
from torchhydro.datasets.unified_data_loader import UnifiedDataLoader


class UnifiedSimulator:
    """
    Organize the simulation process for data.
    We don't do any data processing here, only focus on the simulation aspect
    The data aspect is handled separately in torchhydro.datasets.unified_data_loader.
    """

    def __init__(self, model_cfgs: Dict[str, any], base_cfgs: Dict[str, any]):
        self.model_cfgs = model_cfgs
        self.base_cfgs = base_cfgs
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        weight_path = self.model_cfgs["weight_path"]
        model_name = self.model_cfgs["model_name"]
        model = pytorch_model_dict[model_name](**self.model_cfgs["model_hyperparam"])
        device_map = "cpu" if self.base_cfgs["device"] == -1 else f"cuda:{self.base_cfgs["device"]}"
        checkpoint = torch.load(weight_path, map_location=device_map)
        model.load_state_dict(checkpoint)
        print("Weights sucessfully loaded")
        return model

    def simulate(self, inputs):
        inputs = torch.from_numpy(inputs).float().to(self.model.device)
        seq_first = self.base_cfgs["seq_first"]
        device = self.model.device
        model = self.model
        variable_length_cfgs = self.base_cfgs["variable_length_cfgs"]
        with torch.no_grad():
            targets, outputs = model_infer(seq_first, device, model, inputs, variable_length_cfgs)
        return targets, outputs
    
class UnifiedTester:
    """
    Unified testing interface for different data sources and models.
    Similar with UnifiedModelSetup in Hydromodel
    """
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.data_loader = UnifiedDataLoader(cfgs)
        self.simulator = UnifiedSimulator(
            cfgs["model_cfgs"], cfgs["base_cfgs"]
        )

    def simulate(self):
        """infer using trained model and unnormalized results"""
        batch_size = self.cfgs["base_cfgs"]["batch_size"]
        test_dataloader = self.data_loader.get_dataloader(batch_size)
        simulator = self.simulator
        with torch.no_grad():
            test_preds = []
            obss = []
            for i, batch in enumerate(
                tqdm(test_dataloader, desc="Model inference", unit="batch")
            ):
                ys, pred = simulator.simulate(batch)
                test_preds.append(pred.cpu())
                obss.append(ys.cpu())
                if i % 100 == 0:
                    torch.cuda.empty_cache()
            pred = torch.cat(test_preds, dim=0).numpy()  
            obs = torch.cat(obss, dim=0).numpy()
        if pred.ndim == 2:
            # TODO: check
            # the ndim is 2 meaning we use an Nto1 mode
            # as lookup table is (basin 1's all time length, basin 2's all time length, ...)
            # params of reshape should be (basin size, time length)
            pred = pred.flatten().reshape(test_dataloader.test_data.y.shape[0], -1, 1)
            obs = obs.flatten().reshape(test_dataloader.test_data.y.shape[0], -1, 1)
        evaluation_cfgs = self.cfgs["evaluation_cfgs"]
        obs_xr, pred_xr = get_preds_to_be_eval(
            test_dataloader,
            evaluation_cfgs,
            pred,
            obs,
        )
        return pred_xr, obs_xr