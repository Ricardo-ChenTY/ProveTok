from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..grid.cells import Cell
from ..types import Issue, Token


@dataclass(frozen=True)
class SplitFeaturizerConfig:
    """State/action featurization config for learned split policies (pp.md §5.2)."""

    max_depth: int = 6


def _infer_volume_shape(tokens: Sequence[Token]) -> Tuple[int, int, int]:
    D = H = W = 0
    for t in tokens or []:
        b = getattr(t, "bounds_voxel", None)
        if not isinstance(b, tuple) or len(b) != 6:
            continue
        try:
            z1 = int(b[1])
            y1 = int(b[3])
            x1 = int(b[5])
        except Exception:
            continue
        D = max(D, z1)
        H = max(H, y1)
        W = max(W, x1)
    return (int(D), int(H), int(W))


def _crosses_midline(tok: Token, *, x_mid: float) -> float:
    b = getattr(tok, "bounds_voxel", None)
    if not isinstance(b, tuple) or len(b) != 6:
        return 0.0
    try:
        x0, x1 = float(b[4]), float(b[5])
    except Exception:
        return 0.0
    return 1.0 if (x0 < float(x_mid) < x1) else 0.0


def _rule_blame_map(tokens: Sequence[Token], issues: Sequence[Issue]) -> Dict[str, Set[str]]:
    """Map token refs to the set of rule_ids that blamed them."""
    token_refs = {str(getattr(t, "ref", t.cell_id)) for t in tokens or []}
    out: Dict[str, Set[str]] = {r: set() for r in token_refs}
    for iss in issues or []:
        rid = str(getattr(iss, "rule_id", ""))
        trace = getattr(iss, "evidence_trace", {}) or {}
        for ref in (trace.get("blame_refs", []) or []):
            if not isinstance(ref, str) or not ref:
                continue
            if ref in out:
                out[ref].add(rid)
    return out


def featurize_split_actions(
    *,
    cells: Sequence[Cell],
    tokens: Sequence[Token],
    issues: Sequence[Issue],
    cfg: SplitFeaturizerConfig = SplitFeaturizerConfig(),
    budget_frac: float = 0.0,
    step_frac: float = 0.0,
) -> Tuple[torch.Tensor, List[Cell]]:
    """Featurize candidate split actions for a single decision point.

    Returns:
        (features, ordered_cells)
        - features: (N, F) float32 tensor
        - ordered_cells: cells aligned with rows in features
    """
    tok_by_cell = {str(t.cell_id): t for t in (tokens or [])}
    cells_sorted = sorted(list(cells or []), key=lambda c: c.id())

    D, H, W = _infer_volume_shape(tokens)
    dz = float(D) if D > 0 else 1.0
    dy = float(H) if H > 0 else 1.0
    dx = float(W) if W > 0 else 1.0
    x_mid = 0.5 * float(W) if W > 0 else 0.0

    blame_rules = _rule_blame_map(tokens, issues)
    n_r1 = sum(1 for i in (issues or []) if str(getattr(i, "rule_id", "")) == "R1")
    n_r2 = sum(1 for i in (issues or []) if str(getattr(i, "rule_id", "")) == "R2")
    n_r3 = sum(1 for i in (issues or []) if str(getattr(i, "rule_id", "")) == "R3")
    n_r4 = sum(1 for i in (issues or []) if str(getattr(i, "rule_id", "")) == "R4")

    feats: List[List[float]] = []
    out_cells: List[Cell] = []
    for c in cells_sorted:
        t = tok_by_cell.get(c.id())
        if t is None:
            continue

        level = float(getattr(t, "level", c.level))
        level_norm = float(level) / float(max(1, int(cfg.max_depth)))

        score = float(getattr(t, "score", 0.0))
        unc = float(getattr(t, "uncertainty", 0.0))

        center = getattr(t, "center_voxel", (0.0, 0.0, 0.0))
        try:
            cz, cy, cx = float(center[0]), float(center[1]), float(center[2])
        except Exception:
            cz, cy, cx = 0.0, 0.0, 0.0
        cz_n = cz / dz
        cy_n = cy / dy
        cx_n = cx / dx

        b = getattr(t, "bounds_voxel", (0, 0, 0, 0, 0, 0))
        try:
            z0, z1, y0, y1, x0, x1 = (float(v) for v in b)
            sz_n = max(0.0, (z1 - z0) / dz)
            sy_n = max(0.0, (y1 - y0) / dy)
            sx_n = max(0.0, (x1 - x0) / dx)
        except Exception:
            sz_n = sy_n = sx_n = 0.0

        ref = str(getattr(t, "ref", t.cell_id))
        rules = blame_rules.get(ref, set())
        is_blame = 1.0 if rules else 0.0
        is_r2 = 1.0 if "R2" in rules else 0.0
        is_r3 = 1.0 if "R3" in rules else 0.0
        is_r4 = 1.0 if "R4" in rules else 0.0
        crosses = float(_crosses_midline(t, x_mid=float(x_mid)))

        feats.append(
            [
                level_norm,
                score,
                unc,
                cz_n,
                cy_n,
                cx_n,
                sz_n,
                sy_n,
                sx_n,
                crosses,
                is_blame,
                is_r2,
                is_r3,
                is_r4,
                float(n_r1),
                float(n_r2),
                float(n_r3),
                float(n_r4),
                float(budget_frac),
                float(step_frac),
            ]
        )
        out_cells.append(c)

    if not feats:
        return torch.zeros((0, 20), dtype=torch.float32), []
    return torch.tensor(feats, dtype=torch.float32), out_cells


class SplitPolicyNet(nn.Module):
    """Lightweight action scorer for split decisions (pointer-style).

    Input: per-action feature vector.
    Output: scalar logit per action (softmax over available actions).
    """

    def __init__(self, in_dim: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (N, F) -> logits: (N,)
        if feats.ndim != 2:
            raise ValueError(f"expected 2D feats (N,F), got shape={tuple(feats.shape)}")
        return self.net(feats).squeeze(-1)


@dataclass(frozen=True)
class DPOSplitPolicyConfig:
    max_depth: int = 6
    require_blame: bool = True
    device: str = "cpu"


class DPOSplitPolicy:
    """Inference-time split policy wrapper for a trained SplitPolicyNet."""

    def __init__(
        self,
        model: SplitPolicyNet,
        *,
        cfg: DPOSplitPolicyConfig = DPOSplitPolicyConfig(),
    ):
        self.model = model
        self.cfg = cfg
        self.model.eval().to(torch.device(str(cfg.device)))

    @torch.no_grad()
    def pick_cell(
        self,
        cells: List[Cell],
        tokens: List[Token],
        issues: List[Issue],
        *,
        budget_frac: float = 0.0,
        step_frac: float = 0.0,
        **_kwargs,
    ) -> Optional[Cell]:
        feats, ordered = featurize_split_actions(
            cells=cells,
            tokens=tokens,
            issues=issues,
            cfg=SplitFeaturizerConfig(max_depth=int(self.cfg.max_depth)),
            budget_frac=float(budget_frac),
            step_frac=float(step_frac),
        )
        if not ordered:
            return None
        device = next(self.model.parameters()).device
        logits = self.model(feats.to(device=device))

        if bool(self.cfg.require_blame):
            # Mask to blamed-only actions.
            blame_mask = feats[:, 10] > 0.0  # is_blame
            if not bool(blame_mask.any().item()):
                return None
            masked = logits.clone()
            masked[~blame_mask.to(device=device)] = float("-inf")
            idx = int(torch.argmax(masked).item())
            return ordered[idx]

        idx = int(torch.argmax(logits).item())
        return ordered[idx]


def dpo_loss(
    *,
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    chosen_idx: int,
    rejected_idx: int,
    beta: float = 0.1,
) -> torch.Tensor:
    """Discrete-action DPO loss on a single candidate set (pp.md §5.2)."""
    if policy_logits.ndim != 1 or ref_logits.ndim != 1:
        raise ValueError("policy_logits/ref_logits must be 1D over candidates")
    if policy_logits.shape != ref_logits.shape:
        raise ValueError(f"shape mismatch: policy={tuple(policy_logits.shape)} ref={tuple(ref_logits.shape)}")
    if not (0 <= int(chosen_idx) < int(policy_logits.shape[0])):
        raise ValueError("chosen_idx out of range")
    if not (0 <= int(rejected_idx) < int(policy_logits.shape[0])):
        raise ValueError("rejected_idx out of range")

    logp = F.log_softmax(policy_logits, dim=0)
    logp0 = F.log_softmax(ref_logits, dim=0)
    adv = (logp[int(chosen_idx)] - logp[int(rejected_idx)]) - (logp0[int(chosen_idx)] - logp0[int(rejected_idx)])
    return -F.logsigmoid(float(beta) * adv)


def load_dpo_split_policy(path: str, *, device: str = "cpu", require_blame: bool = True) -> DPOSplitPolicy:
    """Load a `DPOSplitPolicy` from a `train_split_policy_dpo.py` checkpoint."""
    ckpt = torch.load(str(path), map_location=str(device))
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Invalid checkpoint at {path!r} (expected dict with 'state_dict').")
    cfg = ckpt.get("config", {}) or {}
    hidden_dim = int(cfg.get("hidden_dim", 64))
    max_depth = int(cfg.get("max_depth", 6))
    model = SplitPolicyNet(in_dim=20, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["state_dict"])
    return DPOSplitPolicy(
        model,
        cfg=DPOSplitPolicyConfig(max_depth=max_depth, require_blame=bool(require_blame), device=str(device)),
    )
