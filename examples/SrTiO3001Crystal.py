import multem


def SrTiO3001_crystal(na, nb, nc, ncu, rms3d):

    params = multem.CrystalParameters()
    params.na = na
    params.nb = nb
    params.nc = nc
    params.a = 3.9050
    params.b = 3.9050
    params.c = 3.9050

    occ = 1
    region = 0
    charge = 0

    # Sr = 38, Ti = 22; O = 8
    # Z charge x y z rms3d occupancy region charge
    params.layers = [
        multem.AtomList([
            (38, 0.0, 0.0, 0.0, rms3d, occ, region, charge),
            (8, 0.5, 0.5, 0.0, rms3d, occ, region, charge),
        ]),
        multem.AtomList([
            (8, 0.0, 0.5, 0.5, rms3d, occ, region, charge),
            (8, 0.5, 0.0, 0.5, rms3d, occ, region, charge),
            (22, 0.5, 0.5, 0.5, rms3d, occ, region, charge),
        ]),
    ]

    atoms = multem.crystal_by_layers(params)

    dz = params.c / ncu
    lx = na * params.a
    ly = nb * params.b
    lz = nc * params.c

    return atoms, lx, ly, lz, params.a, params.b, params.c, dz
