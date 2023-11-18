import torch.nn as nn
import torch
import math


class SlimmableLinear1(nn.Module):
    def __init__(self, in_features, out_features):
        super(SlimmableLinear1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.width_mult = 1.0

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # print(self.width_mult)
        return nn.functional.linear(input, self.weight * self.width_mult, self.bias * self.width_mult)


# 定义可调整宽度的卷积层
class SlimmableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super(SlimmableConv, self).__init__()
        self.width_mult = 1.0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # 定义可调整宽度的权重和偏置参数
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size ** 2 // self.groups
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return nn.functional.conv2d(input, self.weight * self.width_mult, self.bias * self.width_mult,
                                    self.stride, self.padding, groups=self.groups)


# 定义可调整宽度的深度可分离卷积层
class SlimmableDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SlimmableDepthwiseSeparableConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.width_mult = 1.0

        # 定义深度可分离卷积层
        self.depthwise_conv = SlimmableConv(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                            groups=in_channels)
        self.pointwise_conv = SlimmableConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.depthwise_conv(x * self.width_mult)
        out = self.pointwise_conv(out)
        return out


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, start=False, end=False):
        super(SlimmableLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.width_mult = 1
        self.in_features = in_features
        self.out_features = out_features
        self.start = start
        self.end = end

    def forward(self, input):
        in_features = self.in_features
        out_features = self.out_features
        if self.start == False:
            in_features = int(self.in_features * self.width_mult)
            if in_features <= 0:
                in_features = 1

        if self.end == False:
            out_features = int(self.out_features * self.width_mult)
            if out_features <= 0:
                out_features = 1
        # print(self.width_mult, self.start, in_features, out_features)

        weight = self.weight[:out_features, :in_features]
        if self.bias is not None:
            bias = self.bias[:out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 start=False, end=False):
        super(SlimmableConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                              dilation=dilation, groups=groups, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width_mult = 1
        self.start = start
        self.end = end
        self.groups = groups

    def forward(self, input):
        # print(self.width_mult)
        in_channels = self.in_channels
        out_channels = self.out_channels
        if self.start == False:
            in_channels = int(self.in_channels * self.width_mult)
            if in_channels <= 0:
                in_channels = 1

        if self.end == False:
            out_channels = int(self.out_channels * self.width_mult)
            if out_channels <= 0:
                out_channels = 1
        # print(in_channels, out_channels)

        weight = self.weight[:out_channels, :in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        return nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class SlimmableBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False, start=False,
                 end=False):
        super(SlimmableBatchNorm2d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.width_mult = 1.0
        self.num_features = num_features
        self.start = start
        self.end = end

    def forward(self, input):
        num_features = self.num_features
        if self.start == False:
            num_features = int(self.num_features * self.width_mult)
            if num_features <= 0:
                num_features = 1

        weight = self.weight[:num_features]
        bias = self.bias[:num_features]
        if self.track_running_stats:
            running_mean = self.running_mean[:num_features]
            running_var = self.running_var[:num_features]
        else:
            running_mean = self.running_mean
            running_var = self.running_var

        return nn.functional.batch_norm(
            input, running_mean, running_var, weight, bias,
            self.training or not self.track_running_stats,
            self.momentum, self.eps)
