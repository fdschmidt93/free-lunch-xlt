import torch.nn as nn
from torch import Tensor


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features)
