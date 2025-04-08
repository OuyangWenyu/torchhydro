from torchhydro.datasets.data_sets import BaseDataset
from torchhydro.datasets.data_utils import (
    wrap_t_s_dict,
)

class NarxDataset(BaseDataset):
    """
    a dataset for Narx model.
    """
    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        """
        Initialize the Narx dataset.  for fr in camels.
        Parameters
        ----------
        data_cfgs
        is_tra_val_te
        """
        super(NarxDataset, self).__init__(data_cfgs, is_tra_val_te)
        self.data_cfgs = data_cfgs
        if is_tra_val_te in {"train", "valid", "test"}:
            self.is_tra_val_te = is_tra_val_te
        else:
            raise ValueError(
                "'is_tra_val_te' must be one of 'train', 'valid' or 'test' "
            )
        self.b_nestedness = self.data_cfgs["b_nestedness"]
        # load and preprocess data
        self._load_data()

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.num_samples if self.train_mode else self.ngrid

    def _pre_load_data(self):
        self.train_mode = self.is_tra_val_te == "train"
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)
        self.rho = self.data_cfgs["forecast_history"]
        self.warmup_length = self.data_cfgs["warmup_length"]
        self.horizon = self.data_cfgs["forecast_length"]
        self.b_nestedness = self.data_cfgs["b_nestedness"]

    def _load_data(self):
        self._pre_load_data()
        self._read_xyc()
        # normalization
        norm_x, norm_y, norm_c = self._normalize()
        self.x, self.y, self.c = self._kill_nan(norm_x, norm_y, norm_c)
        self._trans2nparr()
        self._create_lookup_table()

    def _read_xyc_specified_time(self, start_date, end_date):
        """Read x, y, c data from data source with specified time range
        We set this function as sometimes we need adjust the time range for some specific dataset,
        such as seq2seq dataset (it needs one more period for the end of the time range)

        Parameters
        ----------
        start_date : str
            start time
        end_date : str
            end time
        """
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            self.data_cfgs["relevant_cols"],
        )
        # y
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            self.data_cfgs["target_cols"],
        )
        if isinstance(data_output_ds_, dict) or isinstance(data_forcing_ds_, dict):
            data_forcing_ds_ = data_forcing_ds_[list(data_forcing_ds_.keys())[0]]
            data_output_ds_ = data_output_ds_[list(data_output_ds_.keys())[0]]
        data_forcing_ds, data_output_ds = self._check_ts_xrds_unit(
            data_forcing_ds_, data_output_ds_
        )
        if self.b_nestedness:
            nestedness_info = self.data_source.read_nestedness_csv(self.t_s_dict["sites_id"])
        # n   nestedness  streamflow  a forcing type
        data_nested_ds = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            self.data_cfgs["target_cols"],
            all_number=True,
        )
        self.x_origin, self.y_origin, self.c_origin = self._to_dataarray_with_unit(
            data_forcing_ds, data_nested_ds, data_output_ds
        )
