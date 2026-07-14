import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


class DropEdge:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, A, training=True):
        if not training or self.p <= 0:
            return A
        mask = (torch.rand_like(A) > self.p).float()
        mask = torch.triu(mask, 1)
        mask = mask + mask.T
        mask = mask + torch.eye(A.size(0), device=A.device)
        mask = mask.clamp(0, 1)
        return A * mask


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, A_norm):
        support = torch.matmul(x, self.weight)
        out = torch.matmul(A_norm, support)
        return out


class SpatialAttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.Tanh(),
            nn.Linear(in_dim // 4, 1),
        )

    def forward(self, x):
        attn_weights = self.attn(x).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return out, attn_weights


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y
