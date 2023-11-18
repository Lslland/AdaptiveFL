import torch.nn as nn
from sdfl_api.models.slimmable_ops import SlimmableConv2d, SlimmableLinear, SlimmableBatchNorm2d


def conv_block(in_channels, out_channels, pool=False, start=False):
    layers = [SlimmableConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, start=start),
              SlimmableBatchNorm2d(out_channels, track_running_stats=False),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SlimmableResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64, start=True)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        if num_classes == 200:
            self.classifier = nn.Sequential(nn.MaxPool2d(8),
                                            nn.Flatten(),
                                            SlimmableLinear(512, num_classes, bias=False, end=True))
        else:
            self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                            nn.Flatten(),
                                            SlimmableLinear(512, num_classes, bias=False, end=True))

    def set_width_mult(self, width_mult):
        self.width_mult = width_mult
        self.apply(lambda m: setattr(m, 'width_mult', width_mult))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
