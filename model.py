from clize import run
from torch.nn import init
from torch.nn.init import xavier_uniform
import torch
import time
import os
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from data import Dataset

class Sin(nn.Module):

    def forward(self, x):
        return torch.sin(30*x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname == 'Linear':
        xavier_uniform(m.weight.data)
        m.weight.data *= 10
        m.bias.data.fill_(0)

class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim),
            Sin(),
        )
        self.apply(weights_init)

    def forward(self, input):
        x = self.fc(input)
        return x


class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.conv = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 4, 2, 1),
            #nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 1024),
            #nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
        )
        self.apply(weights_init)

    def forward(self, input):
        x = self.conv(input)
        x = x.mean(2)
        x = x.view(x.size(0), 128)
        x = self.fc(x)
        return x
