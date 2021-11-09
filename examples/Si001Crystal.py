import multem


def Si001_crystal(na, nb, nc, ncu, rms3d):

    params = multem.CrystalParameters()
    params.na = na
    params.nb = nb
    params.nc = nc
    params.a = 5.4307
    params.b = 5.4307
    params.c = 5.4307

    occ = 1
    region = 0
    charge = 0

    # Sr = 38, Ti = 22; O = 8
    # Z charge x y z rms3d occupancy region charge
    params.layers = [
        multem.AtomList(
            [
                (14, 0.00, 0.00, 0.00, rms3d, occ, region, charge),
                (14, 0.50, 0.50, 0.00, rms3d, occ, region, charge),
            ]
        ),
        multem.AtomList(
            [
                (14, 0.25, 0.25, 0.25, rms3d, occ, region, charge),
                (14, 0.75, 0.75, 0.25, rms3d, occ, region, charge),
            ]
        ),
        multem.AtomList(
            [
                (14, 0.00, 0.50, 0.50, rms3d, occ, region, charge),
                (14, 0.50, 0.00, 0.50, rms3d, occ, region, charge),
            ]
        ),
        multem.AtomList(
            [
                (14, 0.25, 0.75, 0.75, rms3d, occ, region, charge),
                (14, 0.75, 0.25, 0.75, rms3d, occ, region, charge),
            ]
        ),
    ]

    atoms = multem.crystal_by_layers(params)

    dz = params.c / ncu
    lx = na * params.a
    ly = nb * params.b
    lz = nc * params.c

    return atoms, lx, ly, lz, params.a, params.b, params.c, dz
