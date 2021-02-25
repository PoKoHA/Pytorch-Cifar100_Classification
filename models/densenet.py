import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_CH, growth_rate):
        super().__init__()

        inner_CH = 4 * growth_rate # 앞서 말한 이유로 * 4 , 채널 키움

        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_CH), # pre-activation
            nn.ReLU(inplace=True),
            nn.Conv2d(in_CH, inner_CH, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_CH),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_CH, growth_rate, kernel_size=3, padding=1, bias=False)
            # 다시 채널 축소
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

class Transition(nn.Module):
    def __init__(self, in_CH, out_CH):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_CH),
            nn.Conv2d(in_CH, out_CH, 1, bias=False),
            nn.AvgPool2d(2, stride=2) # 사이즈 크기 감소
        )

    def forward(self, x):
        return self.down_sample(x)

# B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
# C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    # nblocks= num of block , block = bottleneck
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate

        inner_channels = 2 * growth_rate # (?)

        #처음 기본 Conv
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)
        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            # 현재 모듈에서 새로운가 모듈 추가(네임, 모듈)
            self.features.add_module("dense_block_layer_{}".format(index),
                                     self._make_dense_layers(block, inner_channels, nblocks[index]))
            # makse dense layer( block, 인풋, 리스트에 반복횟수)
            inner_channels += growth_rate * nblocks[index] # Growth gate 등차
            out_channels = int(reduction * inner_channels) # 앞서 말한 델타 곱소 채널감소 , transition
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels)) # 크기 감소
            inner_channels = out_channels #

        # 마지막 블
        self.features.add_module("dense_block{}".format(len(nblocks) - 1),
                                 self._make_dense_layers(block, inner_channels, nblocks[len(nblocks) - 1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = self.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}',format(index),
                                   block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

# DenseNet(block, nblocks, growth_rate=12, reduction=0.5, num_class=100)
def densenet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def densenet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def densenet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def densenet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)
