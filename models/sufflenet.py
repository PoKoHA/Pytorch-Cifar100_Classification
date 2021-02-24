from functools import partial
# 기존의 함수에서 인자들을 지정해주어 새로운 함수로 만드는 것

import torch
import torch.nn as nn

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        # x
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        # 차원을 하나 키움 ?
        # suppose a conv layer with g groups whose output has g x n channels
        # 먼저 출력채널차원을 (g, n)으로 reshape, trnasposing, flattening it back
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        x = x.transpose(1, 2).contiguous() # 비연속->연속 / copy해서 새로운 메모리
        x = x.view(batchsize, -1, height, width)

        return x
        '''이 부분 다시 보기'''

class DepthwiseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.depthwise(x)

class PointwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, **kwargs),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.pointwise(x)

class ShuffleNetUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stage, stride, groups):
        super().__init__()
        self.bottleneck = nn.Sequential( # 왜 출력 채널에 4하는가?
            PointwiseConv2d(in_channels, int(out_channels / 4), groups=groups),
            nn.ReLU(inplace=True)
        )

        # stage2에서는 groups Conv 적용안했다. 입력채널이 상대적으로 너무 작아서??
        if stage == 2:
            self.bottleneck = nn.Sequential(
                PointwiseConv2d(in_channels, int(out_channels / 4), groups=groups),
                nn.ReLU(inplace=True)
            )
        # 채널들을 셔플 해주는 구간
        self.channel_shuffle = ChannelShuffle(groups)

        self.depthwise = DepthwiseConv2d(
            int(out_channels / 4), int(out_channels / 4), 3, groups=int(out_channels / 4),
            stride=stride, padding=1
        )

        self.expand = PointwiseConv2d(int(out_channels / 4), out_channels, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = self._add
        self.shortcut = nn.Sequential()

        # 논문 : stride을 sufflenet에 적용되는 경우는 2가지 간단한 수정
        # (1) 3x3 avg pooling을 shortcut path에 추가
        # (2) addition을 cancat으로 대체
        # 적은 추가연산량으로 채널의 차원을 쉽게 키울 수 있다.

        if stride != 1 or in_channels != out_channels: # 입/출력 채널이 같이않을때 왜?
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)

            # 여기서 출력채널에 입력채널을 뺴주는 이유
            self.expand = PointwiseConv2d(int(out_channels / 4), out_channels - in_channels,
                                          groups=groups)

            self.fusion = self._cat

    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        shortcut = self.shortcut(x)

        shuffled = self.bottleneck(x)
        shuffled = self.channel_shuffle(shuffled)
        shuffled = self.depthwise(shuffled)
        shuffled = self.expand(shuffled)

        output = self.fusion(shortcut, shuffled)
        output = self.relu(output)

        return output

class ShuffleNet(nn.Module)

    def __init__(self, num_blocks, num_classes=100, groups=3):
        super().__init__()

        # 출력 채널이 저런게 나오는 이유
        if groups == 1:
            out_channels = [24, 144, 288, 576]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]

        self.conv1 = BasicConv2d(3, out_channels[0], 3, padding=1, stride=1)
        self.input_channels = out_channels[0]

        self.stage2 = self._make_stage(
            ShuffleNetUnit, num_blocks[0], out_channels[1], stride=2, stage=2, groups=groups
        )
        self.stage3 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[1],
            out_channels[2],
            stride=2,
            stage=3,
            groups=groups
        )

        self.stage4 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[2],
            out_channels[3],
            stride=2,
            stage=4,
            groups=groups
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_stage(self, block, num_blocks, out_channels, stride, stage, groups):
        '''셔플넷 stage 만들기
        Args:
            block: shuffle unit
            out_channels: 이 stage의 출력 채널 깊이
            stride: 이 stage의 첫번째 block의 stride
        Return:
            a shuffled net stage
        '''
        strides = [stride] + [1] * (num_blocks - 1) #처음에 2를 주고 나머지 1인 이유?
        stage = []

        for stride in strides:
            stage.append(
                # input= out_CH[0]
                block(self.input_channels, out_channels, stride=stride,stage=stage, groups=groups)
            )
            self.input_channels = out_channels
        return nn.Sequential(*stage)

def shufflenet():
    return ShuffleNet([4, 8, 4])
#4, 8, 4 인 이유 ==> 반복 횟수
