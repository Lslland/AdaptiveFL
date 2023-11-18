import torch
from torch.utils.data import random_split


# 定义数据划分方法
def split_dataset(dataset, num_clients):
    client_data = []
    data_size = len(dataset) // num_clients

    for _ in range(num_clients - 1):
        client_part, dataset = random_split(dataset, [data_size, len(dataset) - data_size])
        client_data.append(client_part)

    client_data.append(dataset)
    return client_data

def count_communication_params(parameters):
    num_non_zero_weights = 0
    for name in parameters:
        num_non_zero_weights += torch.count_nonzero(parameters[name])
    mb_size = num_non_zero_weights * 4 / (1024 ** 2) # MB
    return num_non_zero_weights, float(mb_size)

def count_communication_params_channel(num_non_zero_weights):
    mb_size = num_non_zero_weights * 4 / (1024 ** 2) # MB
    return num_non_zero_weights, mb_size