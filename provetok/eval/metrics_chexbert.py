from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

try:  # Optional dependency
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
except Exception:  # noqa: BLE001
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore


class MissingCheXbertDependency(RuntimeError):
    pass


@dataclass(frozen=True)
class CheXbertConfig:
    """CheXbert-style clinical efficacy metric config.

    This implementation is intentionally backend-agnostic: it loads a HuggingFace
    `AutoModelForSequenceClassification` and turns its logits into a binary
    label vector per report, then computes micro precision/recall/F1 between
    predicted and reference reports.

    Supported output modes:
    - "sigmoid": logits are treated as independent binary logits.
    - "chexbert4": logits are grouped into 4-way classes per condition.
      Positive is class=1; optionally treat uncertain (class=3) as positive.
    - "auto": choose "chexbert4" when num_labels % 4 == 0, else "sigmoid".

    Notes:
    - This does not ship model weights. Provide `model_path` as a local path or
      a HF model id that is available in your environment.
    """

    model_path: str
    device: str = "cpu"
    batch_size: int = 8
    max_length: int = 512

    output_mode: str = "auto"  # "auto" | "sigmoid" | "chexbert4"
    threshold: float = 0.5  # sigmoid mode
    treat_uncertain_as_positive: bool = False


def _micro_prf(pred: np.ndarray, ref: np.ndarray) -> Tuple[float, float, float]:
    pred_b = np.asarray(pred).astype(bool)
    ref_b = np.asarray(ref).astype(bool)
    if pred_b.shape != ref_b.shape:
        raise ValueError(f"shape mismatch: pred={pred_b.shape} ref={ref_b.shape}")

    tp = int(np.logical_and(pred_b, ref_b).sum())
    fp = int(np.logical_and(pred_b, np.logical_not(ref_b)).sum())
    fn = int(np.logical_and(np.logical_not(pred_b), ref_b).sum())

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


class CheXbertLabeler:
    def __init__(self, cfg: CheXbertConfig):
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise MissingCheXbertDependency("Missing dependency: transformers")
        mp = str(cfg.model_path or "").strip()
        if not mp:
            raise ValueError("CheXbertConfig.model_path is required")

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(mp, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(mp)
        self.model.eval()
        self.model.to(torch.device(str(cfg.device)))

        num_labels = int(getattr(getattr(self.model, "config", None), "num_labels", 0) or 0)
        mode = str(cfg.output_mode).strip().lower()
        if mode == "auto":
            mode = "chexbert4" if (num_labels > 0 and num_labels % 4 == 0) else "sigmoid"
        if mode not in ("sigmoid", "chexbert4"):
            raise ValueError(f"Unsupported output_mode={cfg.output_mode!r} (use auto|sigmoid|chexbert4)")
        self._mode = mode

        if self._mode == "chexbert4":
            if num_labels <= 0 or (num_labels % 4 != 0):
                raise ValueError(
                    f"output_mode=chexbert4 requires num_labels % 4 == 0, got num_labels={num_labels}"
                )
            self._num_conditions = int(num_labels // 4)
        else:
            self._num_conditions = int(num_labels)

    @torch.no_grad()
    def predict_binary(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, int(self._num_conditions)), dtype=bool)

        bs = max(1, int(self.cfg.batch_size))
        out: list[np.ndarray] = []
        for i in range(0, len(texts), bs):
            batch = [str(t or "") for t in texts[i : i + bs]]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.cfg.max_length),
            )
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            if self._mode == "sigmoid":
                probs = torch.sigmoid(logits)
                lab = (probs > float(self.cfg.threshold)).detach().cpu().numpy().astype(bool)
                out.append(lab)
            else:
                # (B, C*4) -> (B, C, 4)
                B, L = logits.shape
                C = int(self._num_conditions)
                if int(L) != int(C * 4):
                    raise ValueError(f"Unexpected logits dim for chexbert4: L={L} vs C*4={C*4}")
                x = logits.view(int(B), int(C), 4)
                state = torch.argmax(x, dim=-1).detach().cpu().numpy().astype(np.int64)  # 0..3
                pos = state == 1
                if bool(self.cfg.treat_uncertain_as_positive):
                    pos = np.logical_or(pos, state == 3)
                out.append(pos.astype(bool))

        return np.concatenate(out, axis=0)


def compute_chexbert_prf(
    pred_text: str,
    ref_text: str,
    *,
    labeler: CheXbertLabeler,
) -> Dict[str, float]:
    pred = labeler.predict_binary([str(pred_text)])[0]
    ref = labeler.predict_binary([str(ref_text)])[0]
    p, r, f1 = _micro_prf(pred, ref)
    return {
        "chexbert_precision": float(p),
        "chexbert_recall": float(r),
        "chexbert_f1": float(f1),
    }
