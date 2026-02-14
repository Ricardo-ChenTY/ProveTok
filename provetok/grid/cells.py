from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import re
from typing import Optional

@dataclass(frozen=True, order=True)
class Cell:
    """Regular grid cell family with dyadic splits (octree-like)."""
    level: int
    ix: int
    iy: int
    iz: int

    def id(self) -> str:
        return f"L{self.level}:{self.ix},{self.iy},{self.iz}"

_CELL_ID_RE = re.compile(r"^L(?P<level>\d+):\s*(?P<x>-?\d+)\s*,\s*(?P<y>-?\d+)\s*,\s*(?P<z>-?\d+)\s*$")
_CELL_ID_LEGACY_RE = re.compile(r"^L(?P<level>\d+):\s*\(\s*(?P<x>-?\d+)\s*,\s*(?P<y>-?\d+)\s*,\s*(?P<z>-?\d+)\s*\)\s*$")


def parse_cell_id(cell_id: str) -> Optional[Cell]:
    """Parse canonical `cell_id` into a Cell.

    Canonical format: `L{level}:{ix},{iy},{iz}` (no parentheses).
    Legacy accepted:   `L{level}:(ix,iy,iz)`.
    """
    m = _CELL_ID_RE.match(cell_id)
    if m is None:
        m = _CELL_ID_LEGACY_RE.match(cell_id)
    if m is None:
        return None
    try:
        return Cell(
            level=int(m.group("level")),
            ix=int(m.group("x")),
            iy=int(m.group("y")),
            iz=int(m.group("z")),
        )
    except Exception:
        return None

def root_cell() -> Cell:
    return Cell(level=0, ix=0, iy=0, iz=0)

def split(cell: Cell) -> List[Cell]:
    l = cell.level + 1
    base_x, base_y, base_z = cell.ix * 2, cell.iy * 2, cell.iz * 2
    out = []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                out.append(Cell(l, base_x + dx, base_y + dy, base_z + dz))
    return out

def cell_bounds(cell: Cell, shape: Tuple[int, int, int]) -> Tuple[slice, slice, slice]:
    """Deterministic phi(cell): maps cell_id -> voxel indices slice in (D,H,W)."""
    D, H, W = shape
    n = 2 ** cell.level

    def axis_slice(size: int, i: int) -> slice:
        start = (size * i) // n
        end = (size * (i + 1)) // n
        return slice(start, max(start + 1, end))

    return axis_slice(D, cell.iz), axis_slice(H, cell.iy), axis_slice(W, cell.ix)


def cell_bounds_int(cell: Cell, shape: Tuple[int, int, int]) -> Tuple[int, int, int, int, int, int]:
    """Integer bounds form of `cell_bounds`.

    Returns:
        (z0, z1, y0, y1, x0, x1) with half-open intervals [start, stop).
    """
    z, y, x = cell_bounds(cell, shape)
    return (
        int(z.start or 0),
        int(z.stop or 0),
        int(y.start or 0),
        int(y.stop or 0),
        int(x.start or 0),
        int(x.stop or 0),
    )


def cell_center_voxel(bounds: Tuple[int, int, int, int, int, int]) -> Tuple[float, float, float]:
    """Compute the voxel-space center (cz, cy, cx) from integer bounds."""
    z0, z1, y0, y1, x0, x1 = (int(x) for x in bounds)
    cz = 0.5 * float(z0 + z1)
    cy = 0.5 * float(y0 + y1)
    cx = 0.5 * float(x0 + x1)
    return (cz, cy, cx)
