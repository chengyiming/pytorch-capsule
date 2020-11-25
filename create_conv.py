
import torch.nn as nn
import torch.nn.init as init

class CreateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(CreateConv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,  # fixme constant
                               stride=stride,
                               padding = padding,
                               bias=bias)
        # init.normal_(self.conv.weight, 0, 0.1)
        # init.constant_(self.conv.bias, 0.1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

# 网上实现的截断正太分布
# def truncated_normal_(tensor, mean=0, std=1):
#     size = tensor.shape
#     tmp = tensor.new_empty(size + (4,)).normal_()
#     valid = (tmp < 2) & (tmp > -2)
#     ind = valid.max(-1, keepdim=True)[1]
#     tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
#     tensor.data.mul_(std).add_(mean)
#     return tensor
