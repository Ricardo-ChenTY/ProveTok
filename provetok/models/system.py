from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
import torch.nn as nn

# NOTE: 这个 system 是“占位总装”。
# 当前 demo 仍使用 provetok/bet + provetok/pcg/ToyPCG + verifier。
# 未来你把 ToyPCG 换成一个 nn.Module，就可以在这里统一管理 forward / loss / decode。

@dataclass
class Output:
    loss: torch.Tensor
    logs: Dict[str, float]
    artifact: Dict[str, Any]

class ProveTokSystem(nn.Module):
    def __init__(self):
        super().__init__()
        # placeholder modules
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, batch: Dict[str, Any]) -> Output:
        # TODO: implement your training objective here
        loss = (self.dummy * 0.0) + torch.tensor(0.0, device=self.dummy.device)
        return Output(loss=loss, logs={"loss": float(loss.item())}, artifact={})
