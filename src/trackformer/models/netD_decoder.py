"""
Track query discriminator
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class NetDDecoder(nn.Module):
    def __init__(self, d_model, context=False):
        super(NetDDecoder, self).__init__()
        self.hidden_dim = d_model
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim//2)
        self.linear2 = nn.Linear(self.hidden_dim//2, 2)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim//2)
        self.context = context

    def forward(self, x):
        x = x.view(-1, self.hidden_dim)
        x = self.linear1(x)
        x = F.dropout(F.relu(x), training=self.training)
        if self.context:
            feat = x
        x = self.linear2(x)
        if self.context:
            return x, feat
        else:
            return x


def build_decoder_discriminator(args):
    discriminator = NetDDecoder(args.hidden_dim)
    return discriminator
