import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from datasets.hw2 import MyDataset, get_dataloader
from models.hw2 import ConvA, ConvB, IndependentPixelCNN, SpatialConvA, SpatialConvB
from utils.hw2 import *


class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, conditional_size=None, color_conditioning=False, **kwargs):
        assert mask_type == "A" or mask_type == "B"
        super().__init__(*args, **kwargs)
        self.conditional_size = conditional_size
        self.color_conditioning = color_conditioning
        self.register_buffer("mask", torch.zeros_like(self.weight))
        self.create_mask(mask_type)
        if self.conditional_size:
            if len(self.conditional_size) == 1:
                self.cond_op = nn.Linear(conditional_size[0], self.out_channels)
            else:
                self.cond_op = nn.Conv2d(conditional_size[0], self.out_channels, kernel_size=3, padding=1)

    def forward(self, input, cond=None):
        batch_size = input.shape[0]
        out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.conditional_size:
            if len(self.conditional_size) == 1:
                # Broadcast across height and width of image and add as conditional bias
                out = out + self.cond_op(cond).view(batch_size, -1, 1, 1)
            else:
                out = out + self.cond_op(cond)
        return out

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, : k // 2] = 1
        self.mask[:, :, k // 2, : k // 2] = 1
        if self.color_conditioning:
            assert self.in_channels % 3 == 0 and self.out_channels % 3 == 0
            one_third_in, one_third_out = self.in_channels // 3, self.out_channels // 3
            if mask_type == "B":
                self.mask[:one_third_out, :one_third_in, k // 2, k // 2] = 1
                self.mask[one_third_out : 2 * one_third_out, : 2 * one_third_in, k // 2, k // 2] = 1
                self.mask[2 * one_third_out :, :, k // 2, k // 2] = 1
            else:
                self.mask[one_third_out : 2 * one_third_out, :one_third_in, k // 2, k // 2] = 1
                self.mask[2 * one_third_out :, : 2 * one_third_in, k // 2, k // 2] = 1
        else:
            if mask_type == "B":
                self.mask[:, :, k // 2, k // 2] = 1


if __name__ == "__main__":
    inc = 3
    outc = 6
    kernel_size = (3, 3)

    my = ConvA(in_channels=inc, out_channels=outc, kernel_size=kernel_size)
    berkley = MaskConv2d("A", in_channels=inc, out_channels=outc, kernel_size=kernel_size, color_conditioning=True)

    print(my.mask)
    print(berkley.mask)
