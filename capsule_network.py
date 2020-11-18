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


from capsule_conv_layer import CapsuleConvLayer
from capsule_layer import CapsuleLayer


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

        self.max_pool = nn.MaxPool2d(3, stride=4, padding=1)
        # images第一个卷积层
        self.images_conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=7, # fixme constant
                               stride=2,
                               padding=3,
                               bias=True)
        # images第二个卷积层
        self.images_conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=9, # fixme constant
                               stride=1,
                               bias=True)

        # corp_images第一个卷积层
        self.corp_images_conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=7,  # fixme constant
                               stride=2,
                               padding= 3,
                               bias=True)
        # corp_images第二个卷积层
        self.corp_images_conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=9,  # fixme constant
                               stride=1,
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
        self.reconstruct0 = nn.Linear(num_output_units*output_unit_size, 400)
        self.reconstruct1 = nn.Linear(400, 32*32)
        self.reconstruct2 = nn.Linear(32*32, reconstruction_size)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # print("size()", self.images_conv1(self.max_pool(x1)).size())
        images_conv2 = self.images_conv2(self.images_conv1(self.max_pool(x1)))
        print("images_conv2:", images_conv2.size())
        corp_images_conv2 = self.corp_images_conv2(self.corp_images_conv1(x2))
        print("corp_images_conv2:", corp_images_conv2.size())
        # 在深度方向进行合并
        merge_images = torch.cat((images_conv2, corp_images_conv2), dim= 1)
        print("merge_images:", merge_images.size())
        return self.digits(self.primary(merge_images))

    def loss(self, images, input, target, size_average=True):
        return self.margin_loss(input, target, size_average) + self.reconstruction_loss(images, input, size_average)

    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)

        # ||vc|| from the paper.
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1)).cuda()
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()

        return L_c

    def reconstruction_loss(self, images, input, size_average=True):
        # Get the lengths of capsule outputs.
        v_mag = torch.sqrt((input**2).sum(dim=2))

        # Get index of longest capsule output.
        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data

        # Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
        batch_size = input.size(0)
        all_masked = [None] * batch_size
        for batch_idx in range(batch_size):
            # Get one sample from the batch.
            input_batch = input[batch_idx]

            # Copy only the maximum capsule index from this batch sample.
            # This masks out (leaves as zero) the other capsules in this sample.
            batch_masked = Variable(torch.zeros(input_batch.size())).cuda()
            batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
            all_masked[batch_idx] = batch_masked

        # Stack masked capsules over the batch dimension.
        masked = torch.stack(all_masked, dim=0)

        # Reconstruct input image.
        masked = masked.view(input.size(0), -1)
        output = self.relu(self.reconstruct0(masked))
        output = self.relu(self.reconstruct1(output))
        output = self.sigmoid(self.reconstruct2(output))
        output = output.view(-1, self.image_channels, self.image_height, self.image_width)

        # Save reconstructed images occasionally.
        if self.reconstructed_image_count % 10 == 0:
            if output.size(1) == 2:
                # handle two-channel images
                zeros = torch.zeros(output.size(0), 1, output.size(2), output.size(3))
                output_image = torch.cat([zeros, output.data.cpu()], dim=1)
            else:
                # assume RGB or grayscale
                output_image = output.data.cpu()
            vutils.save_image(output_image, "reconstruction.png")
        self.reconstructed_image_count += 1

        # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
        # Multiplied by a small number so it doesn't dominate the margin (class) loss.
        error = (output - images).view(output.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=1) * 0.0005

        # Average over batch
        if size_average:
            error = error.mean()

        return error
