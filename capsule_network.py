#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from active_function import Mish
from create_conv import CreateConv



from capsule_conv_layer import CapsuleConvLayer
from capsule_layer import CapsuleLayer
from capsule_upsample_conv_layer import CapsuleUpsampleConvLayer


class CapsuleNetwork(nn.Module):
    def __init__(self,
                 image_width, #  28
                 image_height, # 28
                 image_channels, #1
                 conv_inputs,# 1
                 conv_outputs,#256
                 num_primary_units, #8
                 primary_unit_size,#32*6*6
                 num_output_units,#3
                 output_unit_size):#32
        super(CapsuleNetwork, self).__init__()

        self.reconstructed_image_count = 0

        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height

        self.max_pool = nn.MaxPool2d(4, stride=4)
        # images第一个卷积层
        self.images_conv1 = CreateConv(in_channels=1,
                               out_channels=32,
                               kernel_size=8, # fixme constant
                               stride=2,
                               padding=3,
                               bias=True)
        # images第二个卷积层
        self.images_conv2 = CreateConv(in_channels=32,
                               out_channels=32,
                               kernel_size=9, # fixme constant
                               stride=1,
                               padding= 0,
                               bias=True)

        # corp_images第一个卷积层
        self.corp_images_conv1 = CreateConv(in_channels=1,
                               out_channels=32,
                               kernel_size=8,  # fixme constant
                               stride=2,
                               padding= 3,
                               bias=True)
        # corp_images第二个卷积层
        self.corp_images_conv2 = CreateConv(in_channels=32,
                               out_channels=32,
                               kernel_size=9,  # fixme constant
                               stride=1,
                               padding= 0,
                               bias=True)

        # self.merge_conv = CapsuleConvLayer(in_channels=64,
        #                               out_channels=conv_outputs)

        self.primary = CapsuleLayer(in_units=0,
                                    in_channels=64,
                                    num_units=num_primary_units,
                                    unit_size=primary_unit_size,
                                    use_routing=False)

        self.digits = CapsuleLayer(in_units=num_primary_units,
                                   in_channels=primary_unit_size,
                                   num_units=num_output_units,
                                   unit_size=output_unit_size,
                                   use_routing=True)

        reconstruction_size = image_width * image_height * image_channels
        self.reconstruct0 = nn.Linear(32, 256)
        self.reconstruct1 = CapsuleUpsampleConvLayer(16, 4, 8, 'nearest')
        self.reconstruct2 = CapsuleUpsampleConvLayer(4, 8, 4, 'nearest')
        self.reconstruct3 = CapsuleUpsampleConvLayer(8, 16, 4, 'nearest')
        self.compact_layer = nn.Conv2d(in_channels=16,
                                      out_channels=1,
                                      kernel_size=3,  # fixme constant
                                      stride=1,
                                      padding=1,
                                      bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.mish = Mish()

        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x1, x2):
        images_conv1 = self.images_conv1(self.max_pool(x1))
        images_conv2 = 0.2*self.dropout(self.images_conv2(images_conv1))

        corp_conv1 = self.corp_images_conv1(x2)
        corp_conv2 = 0.8*self.dropout(self.corp_images_conv2(corp_conv1))

        # 在深度方向进行合并
        merge_images = torch.cat((images_conv2, corp_conv2), dim= 1)
        return self.digits(self.primary(merge_images))

    def loss(self, images, input, target, size_average=True):
        return self.margin_loss(input, target, size_average) + 0.0005*self.reconstruction_loss(images, input, size_average)

    def margin_loss(self, input, target, size_average=True):
        # [20, 3, 32, 1]
        batch_size = input.size(0)

        # ||vc|| from the paper.
        # [20, 3, 1, 1]
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))
        # print("input:", input[0])
        # print("target:", target)
        # print("v_mag:", v_mag[0])

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = torch.zeros(1).cuda()
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        # 对三个类求和
        L_c = L_c.sum(dim=1)

        # 求一个batch的平均损失
        # if size_average:
        L_c = L_c.mean()
        # print("margin_loss:", L_c)

        return L_c

    def reconstruction_loss(self, images, input, size_average=True):
        # Get the lengths of capsule outputs.
        # [20, 3, 1]
        v_mag = torch.sqrt((input**2).sum(dim=2))
        # Get index of longest capsule output.
        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data
        # [20, 1]

        # Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
        batch_size = input.size(0)
        # 20

        one_hot_labels = torch.zeros(batch_size, 3).scatter(1, v_max_index.cpu(), 1).unsqueeze(-1)
        # [20, 3, 1]

        masked = torch.matmul(input.squeeze().transpose(1,2), one_hot_labels.cuda())

        # Reconstruct input image.
        masked = masked.view(input.size(0), -1)
        # 32->256
        output = self.reconstruct0(masked)
        # 20 16 4 4
        output = output.view((-1, 16, 4, 4))
        output = self.reconstruct1(output)
        # [32 32 4]

        output = self.reconstruct2(output)
        # [128 128 8]

        output = self.reconstruct3(output)
        # [16 512 512]

        #压缩为一层, 也叫做解码层
        decode = self.sigmoid(self.compact_layer(output))
        # [20, 1, 512, 512]

        # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
        # Multiplied by a small number so it doesn't dominate the margin (class) loss.
        error = (decode - images).view(decode.size(0), -1)
        # print("decode:", torch.max(decode), torch.min(decode))
        # print("images:", torch.max(images), torch.min(images))
        error = error**2
        # error = torch.sum(error, dim=1) * 0.0005

        # Average over batch
        # if size_average:
        error = error.mean()

        # print("reconstruction_loss:", error)

        return error

    def acc(self, input, target):
        # input [20, 3, 32, 1]
        # target [20, 1]
        # Get the lengths of capsule outputs.
        # [20, 3, 1]
        v_mag = torch.sqrt((input ** 2).sum(dim=2))
        # Get index of longest capsule output.
        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data
        # [20, 1]
        predicted_class = v_max_index.squeeze()
        correct_prediction = torch.eq(predicted_class, target)
        accuracy = torch.mean(correct_prediction.float())
        return predicted_class, correct_prediction, accuracy




