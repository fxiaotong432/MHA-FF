from functools import partial
from typing import Callable, Union
from typing import Sequence

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers import get_pool_layer
from monai.networks.layers.factories import Conv, Norm

import radiomics
import logging
# from radiomics import featureextractor
from torch.nn import CrossEntropyLoss

import torch.nn as nn
from timm.models.layers import trunc_normal_
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

class InstanceAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(InstanceAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Tanh(),
            nn.Linear(channel // reduction, 1, bias=False),
        )
        self.ins_att = None

    def forward(self, x, visualize=True):
        # x: (B, N, C)
        B, N, C = x.size()
        att = self.attn(x.view(-1, C))  # (B*N, 1)
        att = att.view(B, N, 1)  # (B, N, 1)
        att = torch.softmax(att, dim=1)
        self.ins_att = att
        att = torch.transpose(att, 2, 1)  # (B, 1, N)
        x = att.bmm(x)  # (B, 1, C)
        x = x.squeeze(1)  # (B, C)
        return x
class InstanceAttention_abo(nn.Module):
    def __init__(self, channel, reduction=16):
        super(InstanceAttention_abo, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Tanh(),
            nn.Linear(channel // reduction, 1, bias=False),
        )
        self.ins_att = None

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        B, N, C = x.size()
        att = self.attn(x.view(-1, C))  # (B*N, 1)
        att = att.view(B, N, 1)  # (B, N, 1)
        att = torch.softmax(att, dim=1)
        att = torch.ones([B, N, 1])/N
        att = att.to('cuda:0')
        if visualize:
            self.ins_att = att
        att = torch.transpose(att, 2, 1)  # (B, 1, N)
        x = att.bmm(x)  # (B, 1, C)
        x = x.squeeze(1)  # (B, C)
        return x
class InstanceAttention_2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(InstanceAttention_2, self).__init__()
        # linear
        self.linear = nn.Linear(channel, channel)
        # layer norm
        self.norm = nn.LayerNorm([channel])
        self.attn = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Tanh(),
            nn.Linear(channel // reduction, 1, bias=False),
        )
        self.ins_att = None

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        B, N, C = x.size()
        att = self.attn(x.view(-1, C))  # (B*N, 1)
        att = att.view(B, N, 1)  # (B, N, 1)
        att = torch.softmax(att, dim=1)
        if visualize:
            self.ins_att = att
        att = torch.transpose(att, 2, 1)  # (B, 1, N)
        x = self.norm(self.linear(x))
        x = att.bmm(x)  # (B, 1, C)
        x = x.squeeze(1)  # (B, C)
        return x

class InstanceAttentionBias(nn.Module):
    def __init__(self, channel, reduction=16):
        super(InstanceAttentionBias, self).__init__()
        
        self.attn = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.Tanh(),
            nn.Linear(channel // reduction, 1),
        )
        self.ins_att = None

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        B, N, C = x.size()
        att = self.attn(x.view(-1, C))  # (B*N, 1)
        att = att.view(B, N, 1)  # (B, N, 1)
        att = torch.softmax(att, dim=1)
        if visualize:
            self.ins_att = att
        att = torch.transpose(att, 2, 1)  # (B, 1, N)
        x = att.bmm(x)  # (B, 1, C)
        x = x.squeeze(1)  # (B, C)
        return x


class InstanceAttentionSigmoid(nn.Module):
    def __init__(self, channel, reduction=16):
        super(InstanceAttentionSigmoid, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.Tanh(),
            nn.Linear(channel // reduction, 1),
        )
        self.ins_att = None

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        B, N, C = x.size()
        att = self.attn(x.view(-1, C))  # (B*N, 1)
        att = att.view(B, N, 1)  # (B, N, 1)
        att = torch.softmax(att, dim=1)
        if visualize:
            self.ins_att = att
        att = torch.transpose(att, 2, 1)  # (B, 1, N)
        x = att.bmm(x)  # (B, 1, C)
        x = x.squeeze(1)  # (B, C)
        return x


class InstanceGatedAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(InstanceGatedAttention, self).__init__()
        self.attn_u = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.Tanh(),
        )
        self.attn_v = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.Sigmoid(),
        )
        self.attn = nn.Linear(channel // reduction, 1)
        self.ins_att = None

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        B, N, C = x.size()
        att_u = self.attn_u(x.view(-1, C))  # (B*N, C/r)
        att_v = self.attn_v(x.view(-1, C))  # (B*N, C/r)
        att = self.attn(att_u*att_v)  # (B*N, 1)
        att = att.view(B, N, 1)  # (B, N, 1)
        att = torch.softmax(att, dim=1)
        if visualize:
            self.ins_att = att
        att = torch.transpose(att, 2, 1)  # (B, 1, N)
        x = att.bmm(x)  # (B, 1, C)
        x = x.squeeze(1)  # (B, C)
        return x


class MIL(nn.Module):

    def __init__(self, channel, num_classes=7, att_type='normal', need_cls=True):
        super(MIL, self).__init__()
        self.need_cls = need_cls
        if att_type == 'normal':
            self.ins_attn = InstanceAttention(channel, reduction=16)
        if att_type == 'normal2':
            self.ins_attn = InstanceAttention_2(channel, reduction=16)
        if att_type == 'sigmoid':
            self.ins_attn = InstanceAttentionSigmoid(channel, reduction=16)
        if att_type == 'normnal_with_bias':
            self.ins_attn = InstanceAttentionBias(channel, reduction=16)
        if att_type == 'abo':
            self.ins_attn = InstanceAttention_abo(channel, reduction=16)
        elif att_type == 'gated':
            self.ins_attn = InstanceGatedAttention(channel, reduction=16)
        self.ins_atts = None
        
        if self.need_cls:
            self.fc = nn.Linear(channel, num_classes)

    def forward(self, x, visualize=False):
        # x: (B, N, C)
        x = self.ins_attn(x, visualize)
        if not self.need_cls:
            return x
        if visualize:
            self.ins_atts = x
        x = self.fc(x)
        return x