from torchhydro.configs.config import default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate


def test_dpl4sac(dpl4sac_args):
    cfg = default_config_file()
    update_cfg(cfg, dpl4sac_args)
    train_and_evaluate(cfg)
    print("All processes are finished!")
