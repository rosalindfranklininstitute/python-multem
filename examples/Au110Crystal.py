import multem
from math import sqrt


def Au110_crystal(na, nb, nc, ncu, rms3d):

    params = multem.CrystalParameters()
    params.na = na
    params.nb = nb
    params.nc = nc
    params.a = 4.0782 / sqrt(2)
    params.b = 4.0782
    params.c = 4.0782 / sqrt(2)

    occ = 1
    region = 0
    charge = 0

    # Z charge x y z rms3d occupancy region charge
    params.layers = [
        multem.AtomList(
            [
                (79, 0.00, 0.00, 0.00, rms3d, occ, region, charge),
            ]
        ),
        multem.AtomList(
            [
                (79, 0.50, 0.50, 0.50, rms3d, occ, region, charge),
            ]
        ),
    ]

    atoms = multem.crystal_by_layers(params)

    dz = params.c / ncu
    lx = na * params.a
    ly = nb * params.b
    lz = nc * params.c

    return atoms, lx, ly, lz, params.a, params.b, params.c, dz
