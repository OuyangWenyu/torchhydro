import os
import time

import deepspeed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1000, 5000)
        self.fc2 = nn.Linear(5000, 200)
        self.fc3 = nn.Linear(200, 100)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def simple_test_rank(rank, world_size, dataset):
    # 初始化进程组
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    model = SimpleModel()
    model = model.to(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False, sampler=DistributedSampler(dataset))
    # 打印当前 GPU 的索引
    print(f"Running on rank {rank}.")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        ddp_model.train()
        running_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            # 前向传播
            outputs = ddp_model(inputs).to(rank)
            loss = criterion(outputs, targets)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader)}")
    print("Training complete.")
    # print(f"Result on rank {rank}:\n{result}")
    # 销毁进程组
    dist.destroy_process_group()


def simple_test(model, data_loader, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader)}")
    print("Training complete.")


def data_parallel_difference(process_id):
    # 定义一个简单的神经网络模型
    model = SimpleModel()
    x = torch.randn(100000, 1000)
    y = torch.randn(100000, 100)
    dataset = TensorDataset(x, y)
    # 使用DataParallel包装模型
    if process_id == 'DP':
        data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
        model = nn.DataParallel(model, device_ids=[0, 1, 2])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        simple_test(model, data_loader, device)
    elif process_id == 'DDP':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        world_size = torch.cuda.device_count()
        mp.spawn(simple_test_rank, args=(world_size, dataset), nprocs=world_size, join=True)
    elif process_id == 'DS':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '10010'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
        deepspeed.init_distributed(dist_backend='nccl', auto_mpi_discovery=False)
        data_loader = DataLoader(dataset, batch_size=256, shuffle=False, sampler=DistributedSampler(dataset))
        model_engine = deepspeed.initialize(model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        simple_test(model_engine, data_loader, device)


def test_compare_different_parallel():
    time_0 = time.time()
    data_parallel_difference('DP')
    print('DP time', time.time()-time_0)
    time_1 = time.time()
    data_parallel_difference('DDP')
    print('DDP time', time.time()-time_1)
    '''
    time_2 = time.time()
    data_parallel_difference('DS')
    print('DS time', time.time()-time_2)
    '''
