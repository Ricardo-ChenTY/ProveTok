from provetok.grid.cells import Cell, parse_cell_id


def test_parse_cell_id_roundtrip_canonical():
    cells = [
        Cell(level=0, ix=0, iy=0, iz=0),
        Cell(level=1, ix=1, iy=0, iz=1),
        Cell(level=3, ix=5, iy=6, iz=7),
    ]
    for c in cells:
        parsed = parse_cell_id(c.id())
        assert parsed == c


def test_parse_cell_id_accepts_legacy_parentheses():
    parsed = parse_cell_id("L2:(1,2,3)")
    assert parsed == Cell(level=2, ix=1, iy=2, iz=3)

