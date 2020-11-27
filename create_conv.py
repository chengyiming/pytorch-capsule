import torch.nn as nn
from padding_strategy import Conv2d
from padding_strategy import truncated_normal_
from active_function import Mish

class CreateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(CreateConv, self).__init__()
        if padding == "same":
            self.conv = Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = kernel_size,
                               stride = stride,
                               bias = bias)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding = padding,
                               bias=bias)
        self.conv.weight = truncated_normal_(self.conv.weight, mean=0, std=0.01)

        self.relu = nn.ReLU(inplace=True)
        self.mish = Mish()

    def forward(self, x):
        return self.mish(self.conv(x))

