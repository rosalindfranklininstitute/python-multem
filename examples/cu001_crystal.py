import multem


def cu001_crystal(na, nb, nc, ncu, rms3d):

    params = multem.CrystalParameters()
    params.na = na
    params.nb = nb
    params.nc = nc
    params.a = 3.6150
    params.b = 3.6150
    params.c = 3.6150

    occ = 1
    region = 0
    charge = 0
    # Cu = 29
    # Z charge x y z rms3d occupancy region charge
    params.layers = [
        multem.AtomList([
            (29, 0.0, 0.0, 0.0, rms3d, occ, region, charge),
            (29, 0.5, 0.5, 0.0, rms3d, occ, region, charge),
        ]),
        multem.AtomList([
            (29, 0.0, 0.5, 0.5, rms3d, occ, region, charge),
            (29, 0.5, 0.0, 0.5, rms3d, occ, region, charge),
        ]),
    ]

    atoms = multem.crystal_by_layers(params)

    dz = params.c / ncu
    lx = na * params.a
    ly = nb * params.b
    lz = nc * params.c

    return atoms, lx, ly, lz, params.a, params.b, params.c, dz
