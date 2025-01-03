
## torchrun.py
from tqdm import tqdm
import torch
import torchvision
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
import time
import os
import src.test.deeplearn.googLeNet as googLeNet

# 设置 cuDNN 为确定性算法并禁用自动优化
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

Batch_size = 512
Num_epochs = 50

# 数据准备函数
def prepared_dataloader(world_size, rank):
    # 数据预处理，包括随机裁剪、水平翻转、标准化等
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform)

    # DistributedSampler 确保数据集在多个 GPU 之间被均匀划分
    train_sampler = DistributedSampler(dataset=train_dataset,
                                       num_replicas=world_size,
                                       rank=rank)

    # DataLoader 加载数据，使用 DistributedSampler 保证多机多卡训练时各 GPU 获取不同的数据
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=Batch_size,
                              pin_memory=True,
                              sampler=train_sampler,
                              num_workers=1)
    return train_loader

def train(rank, world_size):
    # 初始化分布式进程组，用于多机多卡训练的调度
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=world_size, 
                            rank=rank)

    # 设置当前进程所使用的 GPU 设备
    local_rank = rank % torch.cuda.device_count()
    print(f'Use GPU: {local_rank} for training.')
    torch.cuda.set_device(local_rank)

    # 准备数据加载器，使用分布式采样器
    train_loader = prepared_dataloader(world_size, rank)

    # 创建模型并将其移动到 GPU 上
    model = googLeNet.GoogLeNet().cuda()

    # 使用 DistributedDataParallel 包裹模型，分发到多个 GPU 进行并行训练
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # 开始训练过程
    for epoch in range(Num_epochs):
        model.train()  # 设置模型为训练模式
        epoch_start_time = time.time()
        running_loss = 0.0

        # 仅在 rank 为 0 的进程中显示进度条
        if rank == 0:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{Num_epochs}')
        else:
            progress_bar = enumerate(train_loader)

        # 遍历数据集
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化一步

            running_loss += loss.item()

            # 如果是主进程，更新进度条显示信息
            if rank == 0:
                progress_bar.set_postfix(loss=loss.item())

        # 计算每个 epoch 的平均损失
        avg_loss = running_loss / len(train_loader)
        epoch_end_time = time.time()

        # 如果是主进程，打印每个 epoch 的时间和平均损失
        if rank == 0:
            print(f'Epoch [{epoch+1}/{Num_epochs}] finished in {(epoch_end_time - epoch_start_time):.2f} seconds.')
            print(f'Epoch [{epoch+1}/{Num_epochs}] Average Loss: {avg_loss:.4f}')

        # 在所有进程之间同步，确保所有进程都完成当前 epoch 才能进入下一轮
        dist.barrier()

    # 训练结束后销毁分布式进程组
    dist.destroy_process_group()

if __name__ == '__main__':
    # 从环境变量中获取进程的 rank 和 world_size，分别表示当前进程的编号和总进程数
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # 启动训练函数
    train(rank, world_size)