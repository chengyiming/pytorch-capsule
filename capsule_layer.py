#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32,  # fixme constant
                               kernel_size=9,  # fixme constant
                               stride=2, # fixme constant
                               bias=True)

    def forward(self, x):
        return self.conv0(x)

class CapsuleLayer(nn.Module):
    # 0 256 8 24*24*32 false
    # 8 24*24*32 3 32 true
    def __init__(self, in_units, in_channels, num_units, unit_size, use_routing):
        super(CapsuleLayer, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.use_routing = use_routing

        if self.use_routing:
            # In the paper, the deeper capsule layer(s) with capsule inputs (DigitCaps) use a special routing algorithm
            # that uses this weight matrix.
            self.W = nn.Parameter(torch.normal(mean=0, std=0.01, size=(1, in_channels, num_units, unit_size, in_units)))
        else:
            self.conv = nn.Conv2d(in_channels=64,
                       out_channels=32*8,
                       kernel_size=9,
                       stride=2,
                       bias=False)

    @staticmethod
    def squash(s):
        s += 0.00001
        # This is equation 1 from the paper.
        # print("squash s.size():", s.size())
        mag_sq = torch.sum(s**2, dim=-2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):

        u = self.conv(x)
        # print("u.size():", u.size())
        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.num_units, -1)
        # print("u.size():", u.size())

        # Return squashed outputs.
        return CapsuleLayer.squash(u)

    def routing(self, x):
        batch_size = x.size(0)

        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)

        # (batch, features, in_units) -> (batch, features, num_units, in_units, 1)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)

        # (batch, features, in_units, unit_size, num_units)
        W = torch.cat([self.W] * batch_size, dim=0)

        # Transform inputs by weight matrix.
        # (batch_size, features, num_units, unit_size, 1)
        u_hat = torch.matmul(W, x)

        # Initialize routing logits to zero.
        b_ij = torch.zeros(1, self.in_channels, self.num_units, 1).cuda()

        # Iterative routing.
        num_iterations = 4
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            # print("b_ij.size():", b_ij.size())
            c_ij = F.softmax(b_ij, dim= 2)

            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # s_j: torch.Size([35, 1, 3, 32, 1])
            # v_j: torch.Size([35, 1, 3, 32, 1])
            # print("s_j:", s_j.size())
            # (batch_size, 1, num_units, unit_size, 1)
            v_j = CapsuleLayer.squash(s_j)
            # print("v_j:", v_j.size())

            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).sum(dim=0, keepdim=True)

            # Update b_ij (routing)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)
