#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from d2l import torch as d2l
import matplotlib.pyplot as plt
from torch.nn.functional import dropout


############################### CV modelling ##################################
class LeNetMy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,6, 5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5), nn.Sigmoid(),
            nn.AvgPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )


if __name__ == '__main__':
    print('hello')

















