#
# multem.__init__.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import numpy
from multem.multem_ext import *  # noqa


def slice_spec_atoms(atoms, length_z, num_slices):
    """
    Slice the atoms into a number of subslices

    Args:
        atoms (list): The atom data
        length_z (float): The size of the sample in Z
        num_slices (int): The number of slices to use

    Yields:
        tuple: (z0, lz, atoms)

    """

    # Check the input
    assert length_z > 0, length_z
    assert num_slices > 0, num_slices

    # The slice thickness
    spec_lz = length_z / num_slices

    # Get the atom z
    _, _, _, atom_z, _, _, _, _ = zip(*atoms)
    atom_z = numpy.array(atom_z)
    print(numpy.max(atom_z), length_z)
    assert numpy.min(atom_z) >= 0
    assert numpy.max(atom_z) <= length_z
    indices = numpy.arange(0, len(atoms))

    # Loop through the slices
    for i in range(num_slices):
        z0 = i * spec_lz
        z1 = (i + 1) * spec_lz
        if i < num_slices - 1:
            selection = (atom_z >= z0) & (atom_z < z1)
        else:
            selection = (atom_z >= z0) & (atom_z <= z1)
        if numpy.count_nonzero(selection) > 0:
            yield (z0, spec_lz, [atoms[i] for i in indices[selection]])
