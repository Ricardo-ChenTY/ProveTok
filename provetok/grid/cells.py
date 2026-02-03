from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True, order=True)
class Cell:
    """Regular grid cell family with dyadic splits (octree-like)."""
    level: int
    ix: int
    iy: int
    iz: int

    def id(self) -> str:
        return f"L{self.level}:{self.ix},{self.iy},{self.iz}"

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
