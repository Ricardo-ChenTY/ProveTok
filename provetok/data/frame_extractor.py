from __future__ import annotations

import re
from dataclasses import asdict
from typing import List, Optional

from ..types import Frame
from ..pcg.schema import FINDINGS, LOCATIONS


_SENT_SPLIT = re.compile(r"[\\n\\.]+")


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().split()).lower()


def _detect_laterality(sent: str) -> str:
    s = sent
    if "bilateral" in s or ("both" in s and "sides" in s):
        return "bilateral"
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return "unspecified"


def _detect_polarity(sent: str) -> str:
    s = sent
    # Simple negation cues.
    if re.search(r"\\bno\\b", s) or re.search(r"\\bwithout\\b", s) or "absent" in s or "negative for" in s:
        return "absent"
    return "present"


def _detect_location(sent: str) -> str:
    s = sent
    # Lobe patterns.
    if "right upper lobe" in s or "rul" in s:
        return "RUL"
    if "right middle lobe" in s or "rml" in s:
        return "RML"
    if "right lower lobe" in s or "rll" in s:
        return "RLL"
    if "left upper lobe" in s or "lul" in s:
        return "LUL"
    if "left lower lobe" in s or "lll" in s:
        return "LLL"
    if "lingula" in s:
        return "lingula"
    # Coarse locations.
    if "pleura" in s or "pleural" in s:
        return "pleura"
    if "heart" in s:
        return "heart"
    if "mediastinum" in s:
        return "mediastinum"
    # Fallback: unspecified
    return "unspecified"


def _size_bin_from_mm(mm: float) -> str:
    x = float(mm)
    if x < 3:
        return "<3mm"
    if x <= 5:
        return "3-5mm"
    if x <= 8:
        return "6-8mm"
    if x <= 20:
        return "9-20mm"
    return ">20mm"


def _detect_size_bin(sent: str) -> str:
    s = sent
    m = re.search(r"(\\d+(?:\\.\\d+)?)\\s*mm", s)
    if not m:
        return "unspecified"
    try:
        mm = float(m.group(1))
    except Exception:
        return "unspecified"
    return _size_bin_from_mm(mm)


def _detect_uncertain(sent: str) -> bool:
    s = sent
    cues = [
        "possible",
        "may represent",
        "cannot exclude",
        "cannot rule out",
        "suggests",
        "likely",
        "suspicious",
    ]
    return any(c in s for c in cues)


class FrameExtractor:
    """Heuristic extractor from free-form report text to bounded finding frames.

    This is intentionally lightweight; the repo primarily uses structured frames
    for training/evaluation. For manifest datasets, this provides a stable
    mapping to the bounded vocab.
    """

    def __init__(self):
        self.findings = [str(x) for x in FINDINGS]

    def extract_frames(self, report_text: str) -> List[Frame]:
        text = _normalize(report_text)
        if not text:
            return []
        sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
        frames: List[Frame] = []

        for sent in sents:
            finding = self._detect_finding(sent)
            if finding is None:
                continue
            polarity = _detect_polarity(sent)
            laterality = _detect_laterality(sent)
            location = _detect_location(sent)
            if location not in set(LOCATIONS):
                location = "unspecified"
            size_bin = _detect_size_bin(sent)
            uncertain = _detect_uncertain(sent)
            # Deterministic confidence heuristic.
            confidence = 0.85 if polarity == "present" else 0.8
            if uncertain:
                confidence = min(confidence, 0.6)

            frames.append(
                Frame(
                    finding=str(finding),
                    polarity=str(polarity),
                    laterality=str(laterality),
                    confidence=float(confidence),
                    location=str(location),
                    size_bin=str(size_bin),
                    severity="unspecified",
                    uncertain=bool(uncertain),
                )
            )

        return frames

    def _detect_finding(self, sent: str) -> Optional[str]:
        s = sent
        # Normalize common aliases.
        if "pleural effusion" in s:
            return "effusion"
        for f in self.findings:
            if f in s:
                return str(f)
        return None


def frames_to_report(frames: List[Frame]) -> str:
    """Deterministic canonical report text from frames (for text-metric evaluation)."""
    lines: List[str] = []
    for fr in frames:
        if fr.polarity in ("absent", "negative"):
            lines.append(f"No {fr.finding}.")
            continue
        lat = "" if fr.laterality in ("unspecified", "") else f"{fr.laterality} "
        loc = "" if fr.location in ("unspecified", "") else f" in the {fr.location}"
        sz = "" if fr.size_bin in ("unspecified", "") else f" ({fr.size_bin})"
        lines.append(f"There is {lat}{fr.finding}{sz}{loc}.")
    return " ".join(lines).strip()
