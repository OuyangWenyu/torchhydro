from torchhydro.configs.config import update_cfg, default_config_file, cmd

def create_config_fabric():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    config_data = default_config_file()
    args = cmd(ctx=[2], strategy='ddp')
    update_cfg(config_data, args)
    return config_data
