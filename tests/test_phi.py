from provetok.grid.cells import root_cell, split, cell_bounds, cell_stable_id

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

def test_stable_id_unique_and_stable():
    c0 = root_cell()
    kids = split(c0)
    ids = [cell_stable_id(c) for c in kids]
    assert len(set(ids)) == len(ids)
    # Stable across calls.
    ids2 = [cell_stable_id(c) for c in kids]
    assert ids == ids2
