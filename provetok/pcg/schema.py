"""Finding frame schema (paper-aligned, bounded claim space).

This file defines the bounded slot vocabularies used by:
- PCG decoding / constrained decoding masks
- Evidence graph construction
- Verifier rules and metrics

The values here are intentionally finite and enumerable.
"""

from .schema_version import SCHEMA_VERSION

# NOTE: Any change to the bounded vocabularies below MUST bump `SCHEMA_VERSION`.

# Minimal but extensible finding vocabulary (can be expanded for real datasets).
FINDINGS = [
    "effusion",
    "nodule",
    "atelectasis",
    "consolidation",
    "pneumothorax",
    "opacity",
    "mass",
    "cardiomegaly",
    "fracture",
    "normal",
]

LATERALITY = ["left", "right", "bilateral", "unspecified"]
POLARITY = ["present", "absent"]

# Chest CT specific (coarse). Keep "unspecified" always allowed.
LOCATIONS = [
    # Lobes
    "RUL",
    "RML",
    "RLL",
    "LUL",
    "LLL",
    "lingula",
    # Coarse
    "right_lung",
    "left_lung",
    "bilateral",
    "mediastinum",
    "heart",
    "pleura",
    # General
    "unspecified",
]

SIZE_BINS = ["<3mm", "3-5mm", "6-8mm", "9-20mm", ">20mm", "unspecified"]
SEVERITY_LEVELS = ["mild", "moderate", "severe", "unspecified"]

# Negation is handled by POLARITY; uncertainty is a separate bounded slot.
UNCERTAINTY = ["certain", "uncertain"]
