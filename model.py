import torch.nn as nn
import torch.nn.functional as F

class LunaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_linear = nn.Linear(1, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.mean(dim=(1, 2, 3, 4)).view(batch_size, 1)
        logits = self.dummy_linear(x_flat)
        return logits, F.softmax(logits, dim=1)