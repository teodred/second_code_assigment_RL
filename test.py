import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


x = torch.randn(1, 6, 8, 8)  # example input tensor with 6 channels

model = nn.Sequential(
    nn.Conv2d(6, 16, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(in_features=576, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=5)
)

for layer in model:
    x = layer(x)
    print(x.shape)