
import torch.nn as nn


class CapsuleUpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, mode):
        super(CapsuleUpsampleConvLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, # fixme constant
                               stride=1,
                               padding= 1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(self.upsample(x)))
