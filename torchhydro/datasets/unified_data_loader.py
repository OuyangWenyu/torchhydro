from torchhydro.datasets.data_dict import datasets_dict
from torchhydro.trainers.train_utils import gnn_collate_fn
from torch.utils.data import DataLoader


class UnifiedDataLoader:
    """
    Unified data loader for different data sources, only FloodEventDatasource is supported for now.
    Functions are listed as follows:
    1. Read Origin Data
    2. Normalize & Denormalize
    """

    def __init__(self, cfgs):
        self.cfgs = cfgs
        self._get_dataset()

    def _get_dataset(self):
        if self.cfgs["data_cfgs"]["dataset"] not in datasets_dict:
            # TODO: support self-made dataset if necessary
            raise NotImplementedError(
                f"Only Dataset in {list(datasets_dict.keys())} are supported, but got {self.cfgs['data_cfgs']['dataset']}"
            )
        if (
            self.cfgs["data_cfgs"]["dataset"] != "FloodEventDataset"
            and self.cfgs["data_cfgs"]["dataset"] != "FloodEventDplDataset"
        ):
            # TODO: support other datasets
            raise NotImplementedError(
                f"Unsupported dataset: {self.cfgs['data_cfgs']['dataset']}"
            )
        self.dataset = datasets_dict[self.cfgs["data_cfgs"]["dataset_name"]](
            self.cfgs, "test"
        )

    def get_dataloader(self, batch_size):
        _collate_fn = None
        if (
            hasattr(self.dataset, "__class__")
            and "GNN" in self.dataset.__class__.__name__
        ):
            _collate_fn = gnn_collate_fn
        return DataLoader(
            self.dataset,
            batch_size,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            collate_fn=_collate_fn,
        )
