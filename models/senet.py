import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicResidualSEBlock(nn.Module):

    expansion = 1 # 언제 쓰는용도(?)

    def __init__(self, in_channels, out_channels, stride, r=16): # 실험적으로 16이 최적
        super().__init__()

        self.residual = nn.Sequential(
            # 기존 resnet basic구조
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        # 이 부분 (????)
        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels *self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        # H x W x C ==> 1 x 1 x C
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # 1 x 1 x c ==> 1 x 1 x c/r ==> 1 x 1 x C
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)

        squeeze = self.squeeze(residual) # squeeze 해줌
        squeeze = squeeze.view(squeeze.size(0), -1) # reshape 어떤 shape(?)
        excitation = self.excitation(squeeze) # reshape 한 squeeze excitation
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)
        # 원래 shape 로 reshape(??)

        x = residual * excitation.expand_as(residual) + shortcut
        '''exapnd_as VS expand
        e.g. tensor([[1],[3],[5]]) ->exapnd(-1,4) ->[[1, 1, 1, 1], [3, ,3, 3, 3], [5,5,5,5]]
             x [1,4,4,5] shape -> y= y.exand_as(x) -> y의 element로 x의 shape맞춤
             expand(*size) -> Tensor
             expand_as(other) -> Tensor
        '''

        return F.relu(x)
    ''' nn.ReLU VS F.relu
    nn.relu : 예를들면 Sequential model에 붙일수 있는 module
    F.relu  : 단지 relu함수를 call하는 API
    나의 코딩 스타일에 따라 편리한 것을 사용
    '''

class BottleneckResidualSEBlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential( # 기존 Bottleneck 구조 1x1, 3x3 1x1
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.BatchNorm2d(out_channels * self.expansion, 1),
            nn.ReLU(inplace=True)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion //r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)

        residual = self.residual(x)
        squueze = self.squeeze(residual)
        squueze = squueze.view(squueze.size(0), -1)
        excitation = self.excitation(squueze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class SEResNet(nn.Module):

    def __init__(self, block, block_num, class_num=100):
        super().__init__()

        self.in_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # _make_stage ( block, num, out_channels, stride)
        self.stage1 = self._make_stage(block, block_num[0], 64, 1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2)
        self.stage4 = self._make_stage(block, block_num[3], 512, 2)

        self.linear = nn.Linear(self.in_channels, class_num)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x


    def _make_stage(self, block, num, out_channels, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        while num - 1:
            layers.append(block(self.in_channels, out_channels, 1))
            num -= 1

        return nn.Sequential(*layers)

def seresnet18():
    return SEResNet(BasicResidualSEBlock, [2, 2, 2, 2])

def seresnet34():
    return SEResNet(BasicResidualSEBlock, [3, 4, 6, 3])

def seresnet50():
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 6, 3])

def seresnet101():
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 23, 3])

def seresnet152():
    return SEResNet(BottleneckResidualSEBlock, [3, 8, 36, 3])
''' resnet의 architecture'''


