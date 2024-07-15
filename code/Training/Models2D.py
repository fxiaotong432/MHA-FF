from functools import partial
from typing import Callable, Union
from typing import Sequence
from MMMIL import *
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


import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from timm.models.layers import trunc_normal_

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            wo_class: bool = False,
            forward_feature: bool = False,
            use_sa: bool = False,
            # in_channel: int = 3,
            in_channel: int = 1,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.wo_class = wo_class
        self.forward_feature = forward_feature
        self.use_sa = use_sa

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        if not forward_feature:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if not wo_class:
            self.fc = nn.Linear(512 * block.expansion *7, num_classes)
        self.features = None
        self.grads = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def save_gradient(self, grad):
        self.grads = grad

    def _forward_impl(self, x: Tensor, visualize, require_grad) -> Tensor:
        if len(x.size()) == 5:
                x = x.reshape(-1, x.size()[2], x.size()[3], x.size()[4])
        B = x.size(0)
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if visualize:
            self.features = x
        if require_grad:
            x.register_hook(self.save_gradient)
        if self.forward_feature:
            return x

        if self.use_sa:
            sp_att = self.spatial_att(x)
            x = x + x * sp_att

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.wo_class:
            return x
        else:
            x = x.view(int(B/7),-1)
            x = self.fc(x)
            return x

    def forward(self, x: Tensor,x_radio: Tensor, visualize=False, require_grad=False) -> Tensor:
        return self._forward_impl(x, visualize, require_grad)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model_dict = model.state_dict()
        model_dict.update(
            {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()})
        model.load_state_dict(model_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


class MultiModalMIL_2d(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, num_mil=4, mixed_mil=False, hidden_dim=128, att_type='normal',with_drop=None, dim_change =False,radio_dim = 70,n_slice = 3,**kwargs):
        super(MultiModalMIL_2d, self).__init__()
        self.mixed_mil = mixed_mil
        self.with_drop = with_drop
        self.dim_change = dim_change
        # resnet 50
        self.ct_backbone = resnet50(pretrained=pretrained, num_classes=num_classes, wo_class=True)
        if self.with_drop:
            self.ct_dropout = torch.nn.Dropout(p=with_drop)
            self.radio_dropout = torch.nn.Dropout(p=with_drop)
        if self.dim_change:
            self.ct_proj = nn.Sequential(
                nn.Linear(512, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.ct_proj = nn.Sequential(
                nn.Linear(2048*n_slice, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        self.radio_proj = nn.Sequential(
            nn.Linear(radio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        if mixed_mil:
            self.ct_mil = MIL(hidden_dim, num_classes, att_type=att_type)
            self.radio_mil = MIL(hidden_dim, num_classes, att_type=att_type)
            init_weight(self.ct_mil)
            init_weight(self.radio_mil)

        self.mils = nn.ModuleList([MIL(hidden_dim, num_classes, att_type=att_type) for _ in range(num_mil)])

        self.ct_atts = None
        self.radio_atts = None
        self.ins_atts = None
        self.mil_scores = []

        init_weight(self.ct_proj)
        init_weight(self.radio_proj)
        for mil in self.mils:
            init_weight(mil)

    def forward(self, x1, x2, visualize=False):
        B1 = x1.size(0)
        N1 = x1.size(1)
        B2 = x2.size(0)
        N2 = x2.size(1)
        v1 = self.ct_backbone(x1,x2,visualize=False, require_grad=False)  # (BN1, C)
        # drop out
        if self.with_drop:
            v1 = self.ct_dropout(v1)
            x2 = self.radio_dropout(x2)
        v1 = v1.view(B1, 1, -1)
        if self.dim_change==True:
            v1 = self.ct_proj(v1.view(B1,4,-1))
        else:
            v1 = self.ct_proj(v1)
        

        v2 = self.radio_proj(x2)
        v2 = v2.view(B2, N2, -1)  # (B, N2, C)
        res = torch.cat([v1, v2], dim=1)  # (B, N1+N2, C)
        x = []
        for mil in self.mils:
            x.append(mil(res, visualize))
        x = torch.stack(x, dim=1)  # (B, num_mil, num_classes)
        x = torch.mean(x, dim=1)  # (B, num_classes)

        return x

class MultiModalMIL_2d_dimchange(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, num_mil=4, mixed_mil=False, hidden_dim=128, att_type='normal',with_drop=None, dim_change =False,radio_dim = 70,n_slice = 3,**kwargs):
        super(MultiModalMIL_2d_dimchange, self).__init__()
        self.mixed_mil = mixed_mil
        self.with_drop = with_drop
        self.dim_change = dim_change
        # resnet 50
        self.ct_backbone = resnet50(pretrained=pretrained, num_classes=num_classes, wo_class=True)
        if self.with_drop:
            self.ct_dropout = torch.nn.Dropout(p=with_drop)
            self.radio_dropout = torch.nn.Dropout(p=with_drop)
        # 都降到hidden_dim维度
        ## 深度学习特征维数
        if self.dim_change:
            self.ct_proj = nn.Sequential(
                nn.Linear(512, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.ct_proj = nn.Sequential(
                nn.Linear(2048*n_slice, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        self.radio_proj = nn.Sequential(
            nn.Linear(radio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        if mixed_mil:
            self.ct_mil = MIL(hidden_dim, num_classes, att_type=att_type)
            self.radio_mil = MIL(hidden_dim, num_classes, att_type=att_type)
            init_weight(self.ct_mil)
            init_weight(self.radio_mil)

        self.mils = nn.ModuleList([MIL(hidden_dim, num_classes, att_type=att_type) for _ in range(num_mil)])

        self.ct_atts = None
        self.radio_atts = None
        self.ins_atts = None
        self.mil_scores = []

        init_weight(self.ct_proj)
        init_weight(self.radio_proj)
        for mil in self.mils:
            init_weight(mil)

    def forward(self, x1, x2, visualize=False):
        B1 = x1.size(0)
        N1 = x1.size(1)
        B2 = x2.size(0)
        N2 = x2.size(1)
        v1 = self.ct_backbone(x1,x2,visualize=False, require_grad=False)  # (BN1, C)
        # drop out
        if self.with_drop:
            v1 = self.ct_dropout(v1)
            x2 = self.radio_dropout(x2)
        if self.dim_change==True:
            v1 = self.ct_proj(v1.view(B1,4,-1))
        else:
            v1 = self.ct_proj(v1)
        
        v1 = v1.view(B1, N1, -1)
        # v1 = v1.view(B1, 1, -1)
        v2 = self.radio_proj(x2)
        v2 = v2.view(B2, N2, -1)  # (B, N2, C)
        res = torch.cat([v1, v2], dim=1)  # (B, N1+N2, C)
        x = []
        for mil in self.mils:
            x.append(mil(res, visualize))
        x = torch.stack(x, dim=1)  # (B, num_mil, num_classes)
        x = torch.mean(x, dim=1)  # (B, num_classes)
        # x = torch.sigmoid(x)
        return x
    
    
    
class MultiModalMIL_2d_dimchange_abl(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, num_mil=4, mixed_mil=False, 
                 hidden_dim=128, att_type='normal', with_drop=None, dim_change=False,
                 radio_dim=70, n_slice=3, ablate_x1=False, ablate_x2=False, **kwargs):
        super(MultiModalMIL_2d_dimchange_abl, self).__init__()
        
        self.ablate_x1 = ablate_x1
        self.ablate_x2 = ablate_x2
        
        self.mixed_mil = mixed_mil
        self.with_drop = with_drop
        self.dim_change = dim_change
        # resnet 50
        self.ct_backbone = resnet50(pretrained=pretrained, num_classes=num_classes, wo_class=True)
        if self.with_drop:
            self.ct_dropout = torch.nn.Dropout(p=with_drop)
            self.radio_dropout = torch.nn.Dropout(p=with_drop)
        if self.dim_change:
            self.ct_proj = nn.Sequential(
                nn.Linear(512, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.ct_proj = nn.Sequential(
                nn.Linear(2048*n_slice, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        self.radio_proj = nn.Sequential(
            nn.Linear(radio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        if mixed_mil:
            self.ct_mil = MIL(hidden_dim, num_classes, att_type=att_type)
            self.radio_mil = MIL(hidden_dim, num_classes, att_type=att_type)
            init_weight(self.ct_mil)
            init_weight(self.radio_mil)

        self.mils = nn.ModuleList([MIL(hidden_dim, num_classes, att_type=att_type) for _ in range(num_mil)])

        self.ct_atts = None
        self.radio_atts = None
        self.ins_atts = None
        self.mil_scores = []

        init_weight(self.ct_proj)
        init_weight(self.radio_proj)
        for mil in self.mils:
            init_weight(mil)

    def forward(self, x1, x2, visualize=False):
        B1 = x1.size(0)
        N1 = x1.size(1)
        B2 = x2.size(0)
        N2 = x2.size(1)
        
        # Ablation: Replace x1 or x2 with random noise if specified
        if self.ablate_x1:
            x1 = torch.randn_like(x1) * x1.std() + x1.mean()
        if self.ablate_x2:
            x2 = torch.randn_like(x2) * x2.std() + x2.mean()
            
        v1 = self.ct_backbone(x1,x2,visualize=False, require_grad=False)  # (BN1, C)
        # drop out
        if self.with_drop:
            v1 = self.ct_dropout(v1)
            x2 = self.radio_dropout(x2)
        if self.dim_change==True:
            v1 = self.ct_proj(v1.view(B1,4,-1))
        else:
            v1 = self.ct_proj(v1)
        
        v1 = v1.view(B1, N1, -1)
        v2 = self.radio_proj(x2)
        v2 = v2.view(B2, N2, -1)  # (B, N2, C)
        res = torch.cat([v1, v2], dim=1)  # (B, N1+N2, C)
        x = []
        for mil in self.mils:
            x.append(mil(res, visualize))
        x = torch.stack(x, dim=1)  # (B, num_mil, num_classes)
        x = torch.mean(x, dim=1)  # (B, num_classes)
        # x = torch.sigmoid(x)
        return x

class MultiModalMIL_2d_ablation(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, num_mil=4, mixed_mil=False, hidden_dim=128, att_type='normal',with_drop=None, dim_change =False,radio_dim = 70,n_slice = 3,**kwargs):
        super(MultiModalMIL_2d_ablation, self).__init__()
        self.mixed_mil = mixed_mil
        self.with_drop = with_drop
        self.dim_change = dim_change
        # resnet 50
        self.ct_backbone = resnet50(pretrained=pretrained, num_classes=num_classes, wo_class=True)
        if self.with_drop:
            self.ct_dropout = torch.nn.Dropout(p=with_drop)
            self.radio_dropout = torch.nn.Dropout(p=with_drop)
        if self.dim_change:
            self.ct_proj = nn.Sequential(
                nn.Linear(512, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.ct_proj = nn.Sequential(
                nn.Linear(2048*n_slice, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        self.radio_proj = nn.Sequential(
            nn.Linear(radio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.fc1=nn.Linear(1024, num_classes)
        if mixed_mil:
            self.ct_mil = MIL(hidden_dim, num_classes, att_type=att_type)
            self.radio_mil = MIL(hidden_dim, num_classes, att_type=att_type)
            init_weight(self.ct_mil)
            init_weight(self.radio_mil)

        self.mils = nn.ModuleList([MIL(hidden_dim, num_classes, att_type=att_type) for _ in range(num_mil)])

        self.ct_atts = None
        self.radio_atts = None
        self.ins_atts = None
        self.mil_scores = []

        init_weight(self.ct_proj)
        init_weight(self.radio_proj)
        for mil in self.mils:
            init_weight(mil)

    def forward(self, x1, x2, visualize=False):
        B1 = x1.size(0)
        N1 = x1.size(1)
        B2 = x2.size(0)
        N2 = x2.size(1)
        v1 = self.ct_backbone(x1,x2,visualize=False, require_grad=False)  # (BN1, C)
        # drop out
        if self.with_drop:
            v1 = self.ct_dropout(v1)
            x2 = self.radio_dropout(x2)
        if self.dim_change==True:
            v1 = self.ct_proj(v1.view(B1,4,-1))
        else:
            v1 = self.ct_proj(v1)
        
        v1 = v1.view(B1, N1, -1)
        v2 = self.radio_proj(x2)
        v2 = v2.view(B2, N2, -1)  # (B, N2, C)
        res = torch.cat([v1, v2], dim=1)  # (B, N1+N2, C)
        res = res.view(B1,-1)
        x=self.fc1(res)
        return x
    
    
    
    
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.,wo_class=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        if wo_class:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )

    def forward(self, img, radio):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
class MultiModalMIL_2d_vit(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, num_mil=4, mixed_mil=False, hidden_dim=128, att_type='normal',with_drop=None, dim_change =False,radio_dim = 70,n_slice = 3,**kwargs):
        super(MultiModalMIL_2d_vit, self).__init__()
        self.mixed_mil = mixed_mil
        self.with_drop = with_drop
        self.dim_change = dim_change
        # resnet 50
        self.ct_backbone = ViT(image_size = 32,
                                patch_size = 16,
                                num_classes = num_classes,
                                dim = 1024,
                                depth = 6,
                                heads = 16,
                                mlp_dim = 2048,
                                dropout = 0.1,
                                emb_dropout = 0.1,
                                wo_class=True,
                                channels=7
                                )
        if self.with_drop:
            self.ct_dropout = torch.nn.Dropout(p=with_drop)
            self.radio_dropout = torch.nn.Dropout(p=with_drop)
        self.ct_proj = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.radio_proj = nn.Sequential(
            nn.Linear(radio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        if mixed_mil:
            self.ct_mil = MIL(hidden_dim, num_classes, att_type=att_type)
            self.radio_mil = MIL(hidden_dim, num_classes, att_type=att_type)
            init_weight(self.ct_mil)
            init_weight(self.radio_mil)

        self.mils = nn.ModuleList([MIL(hidden_dim, num_classes, att_type=att_type) for _ in range(num_mil)])

        self.ct_atts = None
        self.radio_atts = None
        self.ins_atts = None
        self.mil_scores = []

        init_weight(self.ct_proj)
        init_weight(self.radio_proj)
        for mil in self.mils:
            init_weight(mil)

    def forward(self, x1, x2, visualize=False):
        B1 = x1.size(0)
        N1 = x1.size(1)
        B2 = x2.size(0)
        N2 = x2.size(1)
        v1 = self.ct_backbone(x1,x2)  # (BN1, C)
        # drop out
        if self.with_drop:
            v1 = self.ct_dropout(v1)
            x2 = self.radio_dropout(x2)
        if self.dim_change==True:
            v1 = self.ct_proj(v1.view(B1,4,-1))
        else:
            v1 = self.ct_proj(v1)
        
        v1 = v1.view(B1, 1, -1)
        v2 = self.radio_proj(x2)
        v2 = v2.view(B2, 1, -1)  # (B, N2, C)
        res = torch.cat([v1, v2], dim=1)  # (B, N1+N2, C)
        x = []
        for mil in self.mils:
            x.append(mil(res, visualize))
        x = torch.stack(x, dim=1)  # (B, num_mil, num_classes)
        x = torch.mean(x, dim=1)  # (B, num_classes)
        return x