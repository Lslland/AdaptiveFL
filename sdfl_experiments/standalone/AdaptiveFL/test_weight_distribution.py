import torch
import numpy as np
from sdfl_api.models.resnet_9 import SlimmableResNet9


def print_model_parameters(model, weight_threshold_distribution):
    # print number of non-zero parameters
    count_active_weights = 0
    sum_params = 0
    for params in model.parameters():
        sum_params += params.numel()
        temp = torch.abs(params).detach().cpu().numpy()
        count_active_weights += np.size(temp[temp > weight_threshold_distribution])
    print('total:', sum_params, 'threshold:', weight_threshold_distribution, 'activated:', count_active_weights)
    return count_active_weights


def get_weight_distribution(model, device):
    model.to(device)

    params_sum = print_model_parameters(model, 0)
    print('Total parameters' + " " + str(params_sum))

    weight_threshold_distribution = 10
    weight_count_array = []

    while weight_threshold_distribution > 1e-10:
        count_active_weights = print_model_parameters(model, weight_threshold_distribution)
        print(
            'Number of weights' + " " + ">" + str(weight_threshold_distribution) + ": " + str(count_active_weights))
        weight_count_array.append(count_active_weights)
        weight_threshold_distribution = weight_threshold_distribution / 10

    weight_count_density = np.zeros(len(weight_count_array) + 1)
    weight_count_density[0] = weight_count_array[0]
    for i in range(1, len(weight_count_array)):
        weight_count_density[i] = weight_count_array[i] - weight_count_array[i - 1]
    weight_count_density[-1] = params_sum - weight_count_array[-1]
    print('weight_density:', weight_count_density)

if __name__ == '__main__':
    model_path = './weights/resnet9-global-SFW-dynamic-fix-width-0.4-50-NoIID-False.pth'
    # model_path = './weights/resnet9-global-sgd-dynamic-fix-width-0.1-20.pth'
    model = SlimmableResNet9()
    model.load_state_dict(torch.load(model_path))
    get_weight_distribution(model, device=0)
