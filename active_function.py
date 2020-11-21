from torch import nn
import torch
import torch.nn.functional as F

# 定义一些新的激活函数

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded.....")

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x