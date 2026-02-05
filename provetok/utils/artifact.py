from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def try_git_head(repo_root: Path) -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        head = (p.stdout or "").strip()
        return head if head else "unknown"
    except Exception:
        return "unknown"


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        # PyTorch may expose __version__ as a TorchVersion (subclass of str),
        # which is JSON-serializable but can break torch.load(weights_only=True)
        # allowlists. Cast to a plain str for portability.
        "torch": str(getattr(torch, "__version__", "unknown")),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        try:
            idx = int(torch.cuda.current_device())
            props = torch.cuda.get_device_properties(idx)
            info.update(
                {
                    "cuda_device_index": idx,
                    "cuda_device_name": props.name,
                    "cuda_sm": f"{props.major}.{props.minor}",
                    "cuda_total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
        except Exception:
            pass
    return info


def try_manifest_revision(manifest_path: str) -> Tuple[str, str]:
    """Best-effort data revision extraction for manifest-driven datasets.

    Conventions used by `scripts/data/build_*_manifest.py`:
    - `<manifest>.meta.json` contains a `revision` field
    - `<manifest>.splits.json` contains split manifests (train/val/test)
    """
    if not manifest_path:
        return "unknown", ""

    revision = "unknown"
    meta_path = Path(str(manifest_path) + ".meta.json")
    if meta_path.exists():
        try:
            d = json.loads(meta_path.read_text(encoding="utf-8"))
            revision = str(d.get("revision") or revision)
        except Exception:
            pass

    splits_path = Path(str(manifest_path) + ".splits.json")
    split_manifest_path = str(splits_path) if splits_path.exists() else ""
    return revision, split_manifest_path


@dataclass(frozen=True)
class ArtifactMeta:
    timestamp_utc: str
    code_commit: str
    rule_set_version: str
    schema_version: str
    taxonomy_version: str
    data_revision: str
    split_manifest_path: str
    seed: int
    config: Dict[str, Any]
    hardware: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "code_commit": self.code_commit,
            "rule_set_version": self.rule_set_version,
            "schema_version": self.schema_version,
            "taxonomy_version": self.taxonomy_version,
            "data_revision": self.data_revision,
            "split_manifest_path": self.split_manifest_path,
            "seed": self.seed,
            "config": self.config,
            "hardware": self.hardware,
        }


def build_artifact_meta(
    *,
    repo_root: Path,
    seed: int,
    config: Dict[str, Any],
    rule_set_version: str,
    schema_version: str = "unknown",
    taxonomy_version: str = "unknown",
    data_revision: str = "unknown",
    split_manifest_path: str = "",
) -> ArtifactMeta:
    return ArtifactMeta(
        timestamp_utc=utc_now_iso(),
        code_commit=try_git_head(repo_root),
        rule_set_version=str(rule_set_version),
        schema_version=str(schema_version),
        taxonomy_version=str(taxonomy_version),
        data_revision=str(data_revision),
        split_manifest_path=str(split_manifest_path),
        seed=int(seed),
        config=dict(config),
        hardware=hardware_info(),
    )


def save_json(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
