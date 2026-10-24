"""Fixed, jointly optimized ranking plus relation-balanced distance MSE."""
import torch
from torch import nn
from torch.nn import functional as F


def rank_loss(scores, relation, temperature=0.05):
    off = ~torch.eye(len(scores), dtype=torch.bool, device=scores.device)
    values = []
    for depth in (4, 3, 2, 1):
        pos, neg = (relation == depth) & off, (relation < depth) & off
        eligible = pos.any(1) & neg.any(1)
        logits = scores / temperature
        normalizer = torch.logsumexp(logits.masked_fill(~neg, -1e9), dim=1)
        each = F.softplus(normalizer[:, None] - logits)
        query = (each * pos).sum(1) / pos.sum(1).clamp_min(1)
        if eligible.any():
            values.append(query[eligible].mean())
    return torch.stack(values).mean() if values else scores.sum() * 0


class Objective(nn.Module):
    def __init__(self, centers, temperature=0.05, coefficient=20.0):
        super().__init__()
        self.register_buffer('centers', torch.tensor(centers, dtype=torch.float32))
        self.temperature, self.coefficient = temperature, coefficient

    def forward(self, z, codes, a, b, relation):
        scores = z @ z.T
        depths = (codes[:, None] == codes[None, :]).to(torch.int64).cumprod(-1).sum(-1)
        ranking = rank_loss(scores, depths, self.temperature)
        error = (1 - scores[a, b] - self.centers[relation]).square()
        parts = [error[relation == k].mean() for k in range(5) if (relation == k).any()]
        distance = self.coefficient * (torch.stack(parts).mean() if parts else scores.sum() * 0)
        return ranking + distance, ranking, distance
