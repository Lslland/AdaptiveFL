import torch.nn as nn
import torch
from sdfl_api.models.slimmable_ops import SlimmableLinear, SlimmableConv2d

class SlimmableFmnistCNN(nn.Module):
    def __init__(self):
        super(SlimmableFmnistCNN, self).__init__()
        # self.conv1 = SlimmableConv2d(1, 16, kernel_size=5, stride=1, padding=2, start=True)
        # self.conv2 = SlimmableConv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv1 = SlimmableConv2d(1, 8, kernel_size=5, stride=1, padding=2, start=True)
        self.conv2 = SlimmableConv2d(8, 16, kernel_size=5, stride=1, padding=2)
        self.fc1 = SlimmableLinear(in_features=16*7*7, out_features=128)
        self.fc2 = SlimmableLinear(in_features=128, out_features=10, end=True)

    def set_width_mult(self, width_mult):
        self.width_mult = width_mult
        self.apply(lambda m: setattr(m, 'width_mult', width_mult))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x