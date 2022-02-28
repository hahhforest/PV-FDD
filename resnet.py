from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet
import torch.nn.modules.rnn
from torch import Tensor


def conv3x3(inplanes: int,
            outplanes: int,
            stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        inplanes,
        outplanes,
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding=1,
        bias=True
    )


def conv1x1(inplanes: int,
            outplanes: int,
            stride: int = 1) -> nn.Conv2d:
    """1x1卷积，用于短接时降采样"""
    return nn.Conv2d(
        inplanes,
        outplanes,
        kernel_size=(1, 1),
        stride=(stride, stride),
        padding=0,
        bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None) -> None:

        super(BasicBlock, self).__init__()
        # 每次经过此块，通道的扩增倍数
        self.expansion = 1

        self.norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = self.norm_layer(planes)
        # inplace=True, 改变输入数据，节省申请内存开销
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = self.norm_layer(planes)

        # 降采样匹配通道数
        self.downsample = None

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                self.norm_layer(planes)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # 第一层卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二层卷积
        out = self.conv2(out)
        out = self.bn2(out)
        # 降采样以匹配尺寸
        if self.downsample is not None:
            identity = self.downsample(x)
        # 短接
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[BasicBlock],
            num_layers: List[int],
            num_classes: int):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.norm_layer = nn.BatchNorm2d

        # 第一层卷积
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=(7, 7), stride=(2, 2), padding=3,
                               bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # 四个卷积块，每一块由若干残差块组成
        self.layer1 = self._make_layer(block, 64, num_layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_layers[2], stride=2)
        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
            self,
            block: Type[BasicBlock],
            planes: int,
            num_blocks: int,
            stride: int) -> nn.Sequential:

        layers = []
        # 一个高宽减半的ResNet块，步幅为2，后接多个高宽不变的ResNet块，步幅为1
        layers.append(block(self.in_planes, planes, stride))
        for i in range(1, num_blocks):
            layers.append(block(planes, planes, 1))

        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.block1(x)                       # [200, 64, 28, 28]
        x = self.block2(x)                       # [200, 128, 14, 14]
        x = self.block3(x)                       # [200, 256, 7, 7]
        # out = self.block4(out)
        x = F.avg_pool2d(x, 7)                   # [200, 256, 1, 1]
        x = x.view(x.size(0), -1)                # [200,256]
        out = self.outlayer(x)
        return out