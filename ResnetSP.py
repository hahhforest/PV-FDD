from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
import torchvision.models.resnet
import torch.nn.modules.rnn
from torch import Tensor
import scipy.io as scio


def conv3x3_2d(inplanes: int,
               outplanes: int,
               stride: int = 1) -> nn.Conv2d:
    """3x3二维卷积"""
    return nn.Conv2d(
        inplanes,
        outplanes,
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding=1,
        bias=True
    )


def conv1x1_2d(inplanes: int,
               outplanes: int,
               stride: int = 1) -> nn.Conv2d:
    """1x1二维卷积，用于短接时降采样"""
    return nn.Conv2d(
        inplanes,
        outplanes,
        kernel_size=(1, 1),
        stride=(stride, stride),
        padding=0,
        bias=False
    )


def conv3x3_1d(inplanes: int,
               outplanes: int,
               stride: int = 1) -> nn.Conv1d:
    """3x3一维卷积"""
    return nn.Conv1d(
        inplanes,
        outplanes,
        kernel_size=(3,),
        stride=(stride,),
        padding=1,
        bias=True)


def conv1x1_1d(inplanes: int,
               outplanes: int,
               stride: int = 1) -> nn.Conv1d:
    """1x1一维卷积，用于短接时降采样"""
    return nn.Conv1d(
        inplanes,
        outplanes,
        kernel_size=(1,),
        stride=(stride,),
        padding=0,
        bias=False
    )


class BasicBlock2D(nn.Module):
    """二维基本残差块"""

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1) -> None:

        super(BasicBlock2D, self).__init__()
        # 每次经过此块，通道的扩增倍数
        self.expansion = 1

        self.norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3_2d(inplanes, planes, stride)
        self.bn1 = self.norm_layer(planes)
        # inplace=True, 改变输入数据，节省申请内存开销
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_2d(planes, planes, stride=1)
        self.bn2 = self.norm_layer(planes)

        # 降采样匹配通道数
        self.downsample = None

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1_2d(inplanes, planes, stride),
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
        # 降采样以维度匹配：尺寸和通道
        if self.downsample is not None:
            identity = self.downsample(x)
        # 短接
        out += identity
        out = self.relu(out)

        return out


class BasicBlock1D(nn.Module):
    """一维基本残差块"""

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1) -> None:
        super(BasicBlock1D, self).__init__()

        self.expansion = 1

        self.norm_layer = nn.BatchNorm1d
        self.conv1 = conv3x3_1d(inplanes, planes, stride)
        self.bn1 = self.norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_1d(planes, planes, stride=1)
        self.bn2 = self.norm_layer(planes)

        # 降采样匹配通道数
        self.downsample = None

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1_1d(inplanes, planes, stride),
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
        # 降采样以维度匹配：尺寸和通道
        if self.downsample is not None:
            identity = self.downsample(x)
        # 短接
        out += identity
        out = self.relu(out)

        return out


class ResnetSP(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(ResnetSP, self).__init__()

        self.block1_2d = BasicBlock2D(1, 1, 1)
        self.conv1_2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(4, 4), stride=(1, 1), padding=0,
                                  bias=False)

        self.dimension_squeeze = torch.flatten

        self.block1_1d = BasicBlock1D(inplanes=2, planes=2, stride=1)
        self.block2_1d = BasicBlock1D(inplanes=2, planes=4, stride=1)
        self.avgpool1 = nn.AdaptiveAvgPool1d(18)
        self.block3_1d = BasicBlock1D(inplanes=4, planes=4, stride=1)
        self.block4_1d = BasicBlock1D(inplanes=4, planes=8, stride=2)
        self.block5_1d = BasicBlock1D(inplanes=8, planes=8, stride=1)
        self.block6_1d = BasicBlock1D(inplanes=8, planes=16, stride=1)

        self.avgpool2 = nn.AdaptiveAvgPool1d(3)
        self.fc = nn.Linear(3*16, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # C H W
        # 1x40x4
        out = self.block1_2d(x)
        # 2x37x1
        out = self.conv1_2d(out)
        # C W
        # 维度压缩，2x37
        out = self.dimension_squeeze(out, 2)
        # 2x37
        out = self.block1_1d(out)
        # 4x37
        out = self.block2_1d(out)
        # 平均池化，4x18
        out = self.avgpool1(out)
        # 4x18
        out = self.block3_1d(out)
        # 8x9
        out = self.block4_1d(out)
        # 8x9
        out = self.block5_1d(out)
        # 16x9
        out = self.block6_1d(out)
        # 平均池化，16x3
        out = self.avgpool2(out)
        # 维度压缩，1x48
        out = torch.flatten(out, 1)
        # 全连接，num_classes
        out = self.fc(out)

        return out


if __name__ == '__main__':
    model = ResnetSP(9)

    datafile = 'datas/sample_nor.mat'
    data = scio.loadmat(datafile)
    data = data['sample_nor']
    sample = data[0, 0]
    print(sample.shape)

    x = torch.from_numpy(sample)
    print(x.shape)
    x = torch.unsqueeze(x, 0)
    x = torch.unsqueeze(x, 0)
    x = x.float()
    print(x.dtype)

    ans = model.forward(x)
    print(ans.shape)
    print(ans)
