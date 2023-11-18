import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sdfl_api.utils.general import split_dataset


# 加载CIFAR-10数据集
# def load_cifar10_data():
#     transform = transforms.Compose(
#         [transforms.RandomCrop(32, padding=4),
#          transforms.RandomHorizontalFlip(),
#          transforms.ToTensor(),
#          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
#     trainset = torchvision.datasets.CIFAR10(root='../../../data', train=True, download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
#
#     testset = torchvision.datasets.CIFAR10(root='../../../data', train=False, download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
#     return trainloader, testloader


# 加载CIFAR-10数据集
def load_cifar10_data(num_clients, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../../../data', train=True, download=True, transform=transform)

    # 划分数据集给每个客户端
    client_datasets = split_dataset(trainset, num_clients)

    # 创建训练集和测试集的数据加载器
    trainloaders = [DataLoader(client_data, batch_size=batch_size, shuffle=True) for client_data in client_datasets]
    testset = torchvision.datasets.CIFAR10(root='../../../data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloaders, testloader
