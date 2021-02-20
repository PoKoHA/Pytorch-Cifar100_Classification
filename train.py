import os
import sys
import argparse
import time
from _datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR


if __name__ == '__main__': #인터프리티에서 실행 시 이 코드를 돌려라(선언 시작의 의미)
    parser = argparse.ArgumentParser()# 하이퍼파라미트 정의
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()#인자 파싱하기

    net = get_network(args)

    #데이터 전처리
    cifar100_training_loader = get_training_dataloader(
        settings
    )