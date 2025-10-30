from torchhydro.configs.config import default_config_file, cmd


def test_default_config_file():
    config = default_config_file()
    assert isinstance(config, dict)
    assert "model_cfgs" in config
    assert "data_cfgs" in config
    assert "training_cfgs" in config
    assert "evaluation_cfgs" in config


def test_cmd_default_args():
    # Call cmd() directly, which will return the parsed arguments
    args = cmd()

    # Assert default values
    assert args.sub is None
    assert args.source_cfgs is None
    assert args.scaler is None
    assert args.scaler_params is None
    assert args.dataset is None
    assert args.sampler is None
    assert args.fl_sample is None
    assert args.fl_num_users is None
    assert args.fl_local_ep is None
    assert args.fl_local_bs is None
    assert args.fl_frac is None
    assert args.master_addr is None
    assert args.port is None
    assert args.ctx is None
    assert args.rs is None
    assert args.train_mode is None
    assert args.train_epoch is None
    assert args.save_epoch is None
    assert args.save_iter is None
    assert args.loss_func is None
    assert args.loss_param is None
    assert args.train_period is None
    assert args.valid_period is None
    assert args.test_period is None
    assert args.batch_size is None
    assert args.dropout is None
    assert args.warmup_length == 0
