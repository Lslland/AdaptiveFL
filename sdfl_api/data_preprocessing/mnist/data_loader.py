from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from sdfl_api.utils.general import split_dataset


# def load_mnist_data():
#     # 加载MNIST数据集
#     train_dataset = MNIST(root='../../../data', train=True, transform=ToTensor(), download=True)
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     test_dataset = MNIST(root='../../../data', train=False, transform=ToTensor(), download=True)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#     return train_loader, test_loader


def load_mnist_data(num_clients, batch_size):
    train_dataset = MNIST(root='../../../data', train=True, transform=ToTensor(), download=True)
    client_datasets = split_dataset(train_dataset, num_clients)

    # 创建训练集和测试集的数据加载器
    trainloaders = [DataLoader(client_data, batch_size=batch_size, shuffle=True) for client_data in client_datasets]
    testset = MNIST(root='../../../data', train=False, download=True, transform=ToTensor())
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloaders, testloader
