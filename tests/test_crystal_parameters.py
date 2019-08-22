import multem


def test_crystal_parameters():

    crystal_parameters = multem.CrystalParameters()
    crystal_parameters.na = 10
    crystal_parameters.nb = 20
    crystal_parameters.nc = 30
    crystal_parameters.a = 40
    crystal_parameters.b = 50
    crystal_parameters.c = 60
    crystal_parameters.layers = [
        [(1, 2, 3, 4, 5, 6, 7, 8), (9, 10, 11, 12, 13, 14, 15, 16)],
        [(1, 2, 3, 4, 5, 6, 7, 8), (9, 10, 11, 12, 13, 14, 15, 16)],
    ]

    assert crystal_parameters.na == 10
    assert crystal_parameters.nb == 20
    assert crystal_parameters.nc == 30
    assert crystal_parameters.a == 40
    assert crystal_parameters.b == 50
    assert crystal_parameters.c == 60
    assert crystal_parameters.layers == [
        [(1, 2, 3, 4, 5, 6, 7, 8), (9, 10, 11, 12, 13, 14, 15, 16)],
        [(1, 2, 3, 4, 5, 6, 7, 8), (9, 10, 11, 12, 13, 14, 15, 16)],
    ]
