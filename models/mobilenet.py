import torch
import torch.nn as nn

class DepthSeperabelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        # 채널마다 따로 필터 학습
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      groups=in_channels, **kwargs),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Dim reduction위해 많이쓰임 , 채널수 감소 => 연산량 감소
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):

        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class MobileNet(nn.Module):
    '''
    width multipler : 이녀석(a)의 역할은 각각 레이어를 균일하게 얇게 해준다
                      a가 주어졌다면, 입력 채널(M) = aM
                                     출력 채널(N) = aN 가 된다.
    '''

    def __init__(self, width_multiplier=1, class_num=100):
        super().__init__()

        alpha = width_multiplier
        self.stem = nn.Sequential(
            # BasciConv(input채널, output채널, kernel크기, **kwargs)
            BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
            # DepthSeperabelConv2d(입력채널, 출력채널, kernel크기 , **kwargs)
            DepthSeperabelConv2d(
                int(32 * alpha), int(64 * alpha), 3, padding=1, bias=False)
        )

        #downsample
        self.conv1 = nn.Sequential(
            DepthSeperabelConv2d(
                int(64 * alpha), int(128 * alpha), 3, stride=2, padding=1, bias=False
            ),
            DepthSeperabelConv2d(
              int(128 * alpha), int(128 * alpha), 3, padding=1, bias=False
            )
        )

        #downsample
        self.conv2 = nn.Sequential(
            DepthSeperabelConv2d(
                int(128 * alpha), int(256 * alpha), 3, stride=1, padding=1, bias=False
            ),
            DepthSeperabelConv2d(
                int(256 * alpha), int(256 * alpha), 3, stride=1, bias=False)
            )

        #downsample
        self.conv3 = nn.Sequential(
            DepthSeperabelConv2d(
                int(256 * alpha),
                int(512 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),

            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        #downsample
        self.conv4 = nn.Sequential(
            DepthSeperabelConv2d(
                int(512 * alpha), int(1024 * alpha), 3, stride=2, padding=1, bias=False),
            DepthSeperabelConv2d(
                int(1024 * alpha), int(1024 * alpha), 3, padding=1, bias=False
            )
        )

        self.fc = nn.Linear(int(1024 * alpha), class_num)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def mobilenet(alpha=1, class_num=100):
    return MobileNet(alpha, class_num)
