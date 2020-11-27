import torch.nn as nn
from padding_strategy import Conv2d, truncated_normal_
import torch

class CapsuleUpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, mode):
        super(CapsuleUpsampleConvLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.conv = Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, # fixme constant
                               stride=1,
                               padding= "same",
                               bias=True)

        self.conv.weight = truncated_normal_(self.conv.weight, mean=0, std=0.01)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):

        return self.relu(self.conv(self.upsample(x)))
