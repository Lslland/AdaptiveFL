import torch.nn as nn
from sdfl_api.models.slimmable_ops import SlimmableLinear, SlimmableConv2d, SlimmableBatchNorm2d

# 定义可调整宽度的VGG16, VGG11
__all__ = [
    'VGG',
]


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            SlimmableLinear(512, num_classes, end=True),
        )
        if init_weights:
            self._initialize_weights()

    def set_width_mult(self, width_mult):
        self.width_mult = width_mult
        self.apply(lambda m: setattr(m, 'width_mult', width_mult))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.GroupNorm):
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, group_norm=True, dataset='cifar10'):
    layers = []
    in_channels = 3
    start = True
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if start:
                conv2d = SlimmableConv2d(in_channels, v, kernel_size=3, padding=1, start=start)
                start = False
            else:
                conv2d = SlimmableConv2d(in_channels, v, kernel_size=3, padding=1)
            if group_norm:
                # layers += [conv2d, nn.GroupNorm(num_groups=32, num_channels=v), nn.ReLU(inplace=True)]
                # layers += [conv2d, SlimmableBatchNorm2d(v, track_running_stats=False), nn.ReLU(inplace=True)] # cifar-10,100
                if dataset == 'tinyimagenet':
                    layers += [conv2d, SlimmableBatchNorm2d(v, track_running_stats=True), nn.ReLU(inplace=True)] # tiny-imagenet
                else:
                    layers += [conv2d, SlimmableBatchNorm2d(v, track_running_stats=False), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    if dataset == 'tinyimagenet':
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)] # tiny-imagenet
    else:
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def SlimmableVGG11(num_classes, dataset):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = VGG(make_layers(cfg['A'], group_norm=True, dataset=dataset), num_classes=num_classes)

    return model


def SlimmableVGG16(num_classes, dataset):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D'], group_norm=True, dataset=dataset), num_classes)
