import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import IO, Any, BinaryIO, Tuple
from jaxtyping import Bool, Float, Int
from torch import Tensor

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features, bias=False)
        self.ln1 = nn.LayerNorm(out_features)
        self.fc2 = nn.Linear(out_features, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor, verbose: bool = False) -> Tensor:
        if verbose:
            print(f'input dtype: {x.dtype}')
            print(f'fc1 weight dtype: {self.fc1.weight.dtype}')
            print(f'ln1 weight dtype: {self.ln1.weight.dtype}')
            print(f'fc2 weight dtype: {self.fc2.weight.dtype}')

        x = self.fc1(x)
        if verbose:
            print(f'fc1 output dtype: {x.dtype}')

        x = self.relu(x)
        if verbose:
            print(f'relu output dtype: {x.dtype}')

        x = self.ln1(x)
        if verbose:
            print(f'ln1 output dtype: {x.dtype}')

        x = self.fc2(x)
        if verbose:
            print(f'fc2 output dtype: {x.dtype}')
        return x