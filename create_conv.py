
import torch.nn as nn
from padding_strategy import Conv2d

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
                               kernel_size=kernel_size,  # fixme constant
                               stride=stride,
                               padding = padding,
                               bias=bias)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

