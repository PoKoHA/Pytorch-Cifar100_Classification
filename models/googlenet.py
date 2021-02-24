import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1), # 각 레이어에 Norm을 하는  레이어 두어 변형된
                                  # 분호 나오지않도록 함 미니배치마다 한다는 뜻BAtch norm
            nn.ReLU(inplace=True) # inplace : modifiy the input directly
                                  # 메모리 usage 좀 좋아짐 하지만 input을 없앰
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #Factorizing convolutions
        #1개의 5x5 filter 대신 2개의 3x3 conv filter 사용
        #더 적은 파라미터로 같은 receptive field 얻음
        self.b3 = nn.Sequential(
            #Conv2d(in_coannels, out_channels, kernel_size, stride, padding)
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1), # 왜 padding을 1일까
            nn.BatchNorm2d(n5x5, n5x5), # 위에 코드인자1개 여기 2개인 이유
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1), #이부분 생각
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # torch.cat: 연결하는 코드 dim을 통해 어느 차원으로 늘릴것지 판단가능
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class Googlenet(nn.module):
    def __init__(self, num_class=100):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), #input 3채널
            nn.BatchNorm2d(64), #64개를 다 batch
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # although we only use 1 conv layer as prelayer,
        # we still use name a3, b3.......
        # 무슨 의미인지 파악

        # input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj)
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32) # 필터 개수를 어떻게 정하는가?
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        # 일반적 인셉션 서로 쌓인 모듈 모형, 때로는 stride=2풀링을 넣어 grid 절축위해 넣음
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        #input 14x14x480
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size : 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        # 실제로 적용 시 dropout은 삭제해야하는데 이경우에도 삭제?
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        # fc에서 avverage pooling을 쓰는게 성능에 더 좋다는것을 발견되었다 약 0.6%
        # 그리고 dropout은 여전히 남아있다.
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self. linear(x)

def googlenet():
    return Googlenet()
















