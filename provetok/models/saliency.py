from __future__ import annotations

from typing import Union

import torch

from .saliency_cnn3d import SaliencyCNN3D
from .saliency_resunet3d import SaliencyResUNet3D
from .saliency_unet3d import SaliencyUNet3D
from .saliency_vnet3d import SaliencyVNet3D


SaliencyModel = Union[SaliencyCNN3D, SaliencyUNet3D, SaliencyResUNet3D, SaliencyVNet3D]


def _safe_torch_load(path: str, *, map_location: str | torch.device) -> object:
    # PyTorch 2.6 changed `torch.load` default to `weights_only=True`, which can
    # reject checkpoints that include non-allowlisted globals in auxiliary metadata.
    # Prefer safe loading when available, but fall back to full unpickling for
    # backward compatibility with locally-produced artifacts.
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch: no `weights_only` kwarg.
        return torch.load(str(path), map_location=map_location)
    except Exception:
        return torch.load(str(path), map_location=map_location, weights_only=False)


def load_saliency_model(path: str, *, map_location: str | torch.device = "cpu") -> SaliencyModel:
    ckpt = _safe_torch_load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid saliency checkpoint at {path!r}: expected dict, got {type(ckpt)}")

    if "saliency_unet3d" in ckpt:
        payload = ckpt.get("saliency_unet3d")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid saliency checkpoint at {path!r}: 'saliency_unet3d' must be a dict")
        return SaliencyUNet3D.from_checkpoint(payload)

    if "saliency_resunet3d" in ckpt:
        payload = ckpt.get("saliency_resunet3d")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid saliency checkpoint at {path!r}: 'saliency_resunet3d' must be a dict")
        return SaliencyResUNet3D.from_checkpoint(payload)

    if "saliency_vnet3d" in ckpt:
        payload = ckpt.get("saliency_vnet3d")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid saliency checkpoint at {path!r}: 'saliency_vnet3d' must be a dict")
        return SaliencyVNet3D.from_checkpoint(payload)

    if "saliency_cnn3d" in ckpt:
        payload = ckpt.get("saliency_cnn3d")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid saliency checkpoint at {path!r}: 'saliency_cnn3d' must be a dict")
        return SaliencyCNN3D.from_checkpoint(payload)

    keys = ", ".join(sorted(str(k) for k in ckpt.keys()))
    raise ValueError(
        f"Invalid saliency checkpoint at {path!r}: expected one of keys "
        f"{{'saliency_unet3d','saliency_resunet3d','saliency_vnet3d','saliency_cnn3d'}}, got keys=[{keys}]"
    )
