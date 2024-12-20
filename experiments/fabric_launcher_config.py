from torchhydro.configs.config import update_cfg, default_config_file, cmd

def create_config_fabric():
    # 设置测试所需的项目名称和默认配置文件
    config_data = default_config_file()
    # 填充测试所需的命令行参数
    args = cmd(ctx=[0], strategy='fsdp')
    # 更新默认配置
    update_cfg(config_data, args)
    return config_data
