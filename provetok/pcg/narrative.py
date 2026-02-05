from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

from ..types import Frame, Generation


def render_generation_text(gen: Generation) -> str:
    """Deterministic narrative text derived from the findings table (dual-channel output).

    This is intentionally strict and machine-parseable so the verifier can
    round-trip text -> frames/citations/q/refusal without ambiguity.
    """
    lines: List[str] = []
    for idx, f in enumerate(gen.frames):
        cites = gen.citations.get(idx, [])
        q = gen.q.get(idx, 0.0)
        refusal = bool(gen.refusal.get(idx, False))
        cite_str = ",".join(str(int(x)) for x in cites)
        lines.append(
            " ".join(
                [
                    f"FRAME {idx}",
                    f"finding={f.finding}",
                    f"polarity={f.polarity}",
                    f"laterality={f.laterality}",
                    f"location={f.location}",
                    f"size_bin={f.size_bin}",
                    f"severity={f.severity}",
                    f"uncertain={1 if f.uncertain else 0}",
                    f"confidence={repr(float(f.confidence))}",
                    f"q={repr(float(q))}",
                    f"refusal={1 if refusal else 0}",
                    f"cites=[{cite_str}]",
                ]
            )
        )
    return "\n".join(lines)


def parse_generation_text(text: str) -> Tuple[List[Frame], Dict[int, List[int]], Dict[int, float], Dict[int, bool]]:
    frames: List[Frame] = []
    citations: Dict[int, List[int]] = {}
    q: Dict[int, float] = {}
    refusal: Dict[int, bool] = {}

    if not text.strip():
        return frames, citations, q, refusal

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3 or parts[0] != "FRAME":
            raise ValueError(f"Unparseable generation text line: {line!r}")

        idx = int(parts[1])
        kv: Dict[str, str] = {}
        for p in parts[2:]:
            if "=" not in p:
                raise ValueError(f"Unparseable key-value token: {p!r} in line: {line!r}")
            k, v = p.split("=", 1)
            kv[k] = v

        cites_raw = kv.get("cites", "[]").strip()
        if not (cites_raw.startswith("[") and cites_raw.endswith("]")):
            raise ValueError(f"Invalid cites format: {cites_raw!r}")
        inner = cites_raw[1:-1].strip()
        cites = [int(x) for x in inner.split(",") if x.strip() != ""] if inner else []

        frame = Frame(
            finding=kv["finding"],
            polarity=kv["polarity"],
            laterality=kv["laterality"],
            confidence=float(kv["confidence"]),
            location=kv.get("location", "unspecified"),
            size_bin=kv.get("size_bin", "unspecified"),
            severity=kv.get("severity", "unspecified"),
            uncertain=(kv.get("uncertain", "0") == "1"),
        )
        frames.append(frame)
        citations[idx] = cites
        q[idx] = float(kv.get("q", "0.0"))
        refusal[idx] = (kv.get("refusal", "0") == "1")

    return frames, citations, q, refusal


def roundtrip_equal(a: Generation, b: Generation, *, atol: float = 1e-8) -> bool:
    if len(a.frames) != len(b.frames):
        return False
    for fa, fb in zip(a.frames, b.frames):
        if asdict(fa) != asdict(fb):
            # confidence is a float; allow tiny tolerance on this field.
            if (
                fa.finding != fb.finding
                or fa.polarity != fb.polarity
                or fa.laterality != fb.laterality
                or fa.location != fb.location
                or fa.size_bin != fb.size_bin
                or fa.severity != fb.severity
                or fa.uncertain != fb.uncertain
            ):
                return False
            if abs(float(fa.confidence) - float(fb.confidence)) > atol:
                return False
    if a.citations != b.citations:
        return False
    if a.refusal != b.refusal:
        return False
    # q float tolerance
    if set(a.q.keys()) != set(b.q.keys()):
        return False
    for k in a.q.keys():
        if abs(float(a.q[k]) - float(b.q[k])) > atol:
            return False
    return True
