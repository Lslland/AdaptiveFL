import torch.nn as nn
import torch.nn.functional as F
from sdfl_api.models.slimmable_ops import SlimmableConv2d, SlimmableLinear, SlimmableBatchNorm2d

# 定义残差块
class SlimmableBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(SlimmableBasicBlock, self).__init__()
        self.width_mult = 1.0

        self.conv1 = SlimmableConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = SlimmableBatchNorm2d(planes)
        self.conv2 = SlimmableConv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = SlimmableBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                SlimmableConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                SlimmableBatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x * self.width_mult)))
        out = self.bn2(self.conv2(out * self.width_mult))
        out += self.shortcut(x * self.width_mult)
        out = F.relu(out)
        return out


# 定义可调整宽度的ResNet-18
class SlimmableResNet18(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(SlimmableResNet18, self).__init__()
        self.in_planes = 64
        self.width_mult = 1.0

        self.conv1 = SlimmableConv2d(3, 64, kernel_size=3, stride=1, padding=1, start=True)
        self.bn1 = SlimmableBatchNorm2d(64)
        self.layer1 = self._make_layer(SlimmableBasicBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(SlimmableBasicBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(SlimmableBasicBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(SlimmableBasicBlock, 512, num_blocks[3], stride=2)
        self.linear = SlimmableLinear(512 * SlimmableBasicBlock.expansion, num_classes, end=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_width_mult(self, width_mult):
        self.width_mult = width_mult
        self.apply(lambda m: setattr(m, 'width_mult', width_mult))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x * self.width_mult)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out