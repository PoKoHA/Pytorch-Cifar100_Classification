import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    '''
    basic block for resnet18 and resnet 34
    '''
    expansion = 1 # expansion => 필요 시 output 채널을 늘리기 위해서
                  # 사용은 어떻게하는가?

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(# 하나의 세트 느낌
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()
        # shortcut 출력 dim 이랑 residual 이 같지 않을경우
        # Conv 1x1 을 사용하여 dim을 맞춰주자
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    '''
    Residual block for 50이상 layer
    '''

    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, kernel_size = 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=Flase),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))



class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64 # conv1에서 7x7 64 stride 2 이용하므로

        # 원래 7x7 이지만 cifar 이미지가 32x3x로 작아서
        # 3x3 kernel 사용함
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 논문과 다른 inputsize 사용할 것 이므로 conv2_x의 stride = 1
        # 각 레이어에 블락을 몇번 쓸지 리스트로 받음 num_block
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, out_channels, num_blocks, stride):
        '''

        :param block: basic block or bottleneck
        :param out_channels: 출력의 채널
        :param num_blocks: 한 layer에 얼마나 많은 blocks
        :param stride: first block stride
        :return: a resnet layer
        '''

        # first block 1 or 2
        # other  blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            # layer에
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

#표에 나온 것처럼 블락 갯수를 대입
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])

'''요약
 resnet에서 BLOCK 타입과 [] 리스트 식으로 각 레이어에 그 블락 개수를 입력
e.g)ResNet(BasicBlock, [2, 2, 2, 2])

-->Renet --> 논문에서는 imagenet에서 사용한 kenel크기로 7x7 but cifar에서는 3x3kernel

-->conv1 에서 in_channel 3을 받아 out_channel 64 출력

-->	conv2_x로 이동 self._make_layer(block, 64, num_block[0], 1) = (BasicBlock, 64, 2, 1)
	conv3_x(BasicBlock, 128, 2, 2)
	conv4_x(BasicBlock, 256, 2, 2) 식으로 각 레이어 실행


-->make_layer로 이동(block, out채널, num_blocks, stride)인자

--> [stirde] + [1]*(num_blocks -1) == 처음 블락에 2,1 stride 주고 (pooling효과) 나머지 다 1로 줌

---> for strdie in strides: 각 리스트 stride 대입
       layers.append(block(self.in_channels, out_cnannels, stride)) ==>여기서 layer에 Basciblock담음
       입력 채널 = out_channels * block.expansion

return nn.Sequentail(*layers) ==> 모든 model 실행시킴

-->Conv5_x 실행 후 avgpool2d ((1,1)) 실행하여 1x1크기로 만들어줌


--->Linear( 512 * block.expansin, num_classes) 입력으로 아까 출력값 받고 우리가 분류할 class 100
개
'''