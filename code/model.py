"""model.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm/class có `raise NotImplementedError`.

Model: MLP cho bài toán 7 lớp, shape cố định (xem README mục 3 và GUIDE, "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS bên dưới.
"""
from __future__ import annotations

import torch
import torch.nn as nn

EXPECTED_PARAMS = {
    (256, 128): 47_879,
    (512, 256): 161_287,
    (256, 128, 64): 55_687,
}

class MLP(nn.Module):
    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        layers = []
        in_dim = in_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def init_weights(model: nn.Module, init: str) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init == "zeros":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif init == "normal":
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif init == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif init == "default":
                pass

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    model.eval()
    h = x
    stds = []
    for m in model.net:
        h = m(h)
        if isinstance(m, nn.ReLU):
            stds.append(h.std().item())
    return stds
