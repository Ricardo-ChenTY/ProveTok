from provetok.grid.cells import root_cell, split, cell_bounds

def test_phi_nonempty():
    shape = (10, 20, 30)
    c0 = root_cell()
    slc = cell_bounds(c0, shape)
    assert (slc[0].stop - slc[0].start) > 0
    assert (slc[1].stop - slc[1].start) > 0
    assert (slc[2].stop - slc[2].start) > 0

def test_split_8():
    c0 = root_cell()
    kids = split(c0)
    assert len(kids) == 8
