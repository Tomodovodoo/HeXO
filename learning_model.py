"""Small shared pattern network, exported as a 729-entry native residual table."""
import torch
from torch import nn


def pattern_data(device="cpu"):
    codes = torch.arange(729, device=device)
    powers = 3 ** torch.arange(6, device=device)
    digits = codes[:, None] // powers % 3
    swapped = torch.where(digits == 0, 0, 3-digits)
    swap = (swapped * powers).sum(1)
    reverse = (digits.flip(1) * powers).sum(1)
    inputs = nn.functional.one_hot(digits, 3).float().flatten(1)
    counts0, counts1 = (digits == 1).sum(1), (digits == 2).sum(1)
    weights = torch.tensor([0, 1, 12, 150, 2400, 24000, 1000000], device=device)
    baseline = torch.where(counts1 == 0, weights[counts0],
                           torch.where(counts0 == 0, -weights[counts1], 0)).float()
    return inputs, swap, reverse, baseline


class PatternModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = nn.Sequential(nn.Linear(18, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.mapping[-1].weight)
        nn.init.zeros_(self.mapping[-1].bias)

    def table(self, inputs, swap, reverse):
        raw = 300 * self.mapping(inputs).squeeze(1).tanh()
        # Color antisymmetry and line reversal hold before and after quantization.
        return (raw - raw[swap] + raw[reverse] - raw[swap[reverse]]) / 4

    def forward(self, features, inputs, swap, reverse, baseline):
        table = self.table(inputs, swap, reverse)
        return torch.tanh((features @ (baseline + table)) / 6000), table
