import multem
import numpy
import time
from SrTiO3001Crystal import SrTiO3001_crystal


def create_input_multislice(n_phonons, single_phonon_conf=False):

    input_multislice = multem.Input()

    input_multislice.simulation_type = "EWRS"
    input_multislice.interaction_model = "Multislice"
    input_multislice.potential_slicing = "Planes"

    input_multislice.potential_type = "Lobato_0_12"

    input_multislice.pn_model = "Frozen_Phonon"
    input_multislice.pn_dim = 110
    input_multislice.pn_seed = 300183
    input_multislice.pn_single_conf = single_phonon_conf
    input_multislice.pn_nconf = n_phonons

    input_multislice.thick_type = "Whole_Spec"

    na = 12
    nb = 12
    nc = 12
    ncu = 2
    rms3d = 0.085

    (
        input_multislice.spec_atoms,
        input_multislice.spec_lx,
        input_multislice.spec_ly,
        input_multislice.spec_lz,
        a,
        b,
        c,
        input_multislice.spec_dz,
    ) = SrTiO3001_crystal(na, nb, nc, ncu, rms3d)

    input_multislice.nx = 1024
    input_multislice.ny = 1024

    input_multislice.iw_type = "Plane_Wave"
    input_multislice.iw_x = [0.5 * input_multislice.spec_lx]
    input_multislice.iw_y = [0.5 * input_multislice.spec_ly]

    input_multislice.cond_lens_m = 0
    input_multislice.cond_lens_c_10 = 1110
    input_multislice.cond_lens_c_30 = 3.3
    input_multislice.cond_lens_c_50 = 0.00
    input_multislice.cond_lens_c_12 = 0.0
    input_multislice.cond_lens_phi_12 = 0.0
    input_multislice.cond_lens_c_23 = 0.0
    input_multislice.cond_lens_phi_23 = 0.0
    input_multislice.cond_lens_inner_aper_ang = 0.0
    input_multislice.cond_lens_outer_aper_ang = 7.50
    # input_multislice.cond_lens_ti_sigma = 32
    # input_multislice.cond_lens_ti_npts = 10
    # input_multislice.cond_lens_si_sigma = 0.2
    # input_multislice.cond_lens_si_rad_npts = 8
    input_multislice.cond_lens_zero_defocus_type = "First"
    input_multislice.cond_lens_zero_defocus_plane = 0

    input_multislice.obj_lens_m = 0
    input_multislice.obj_lens_c_10 = 15.836
    input_multislice.obj_lens_c_30 = 1e-03
    input_multislice.obj_lens_c_50 = 0.00
    input_multislice.obj_lens_c_12 = 0.0
    input_multislice.obj_lens_phi_12 = 0.0
    input_multislice.obj_lens_c_23 = 0.0
    input_multislice.obj_lens_phi_23 = 0.0
    input_multislice.obj_lens_inner_aper_ang = 0.0
    input_multislice.obj_lens_outer_aper_ang = 24.0
    # input_multislice.obj_lens_ti_sigma = 32
    # input_multislice.obj_lens_ti_npts = 10
    input_multislice.obj_lens_zero_defocus_type = "Last"
    input_multislice.obj_lens_zero_defocus_plane = 0

    input_multislice.output_area_ix_0 = 1
    input_multislice.output_area_iy_0 = 1
    input_multislice.output_area_ix_e = 1
    input_multislice.output_area_iy_e = 1

    return input_multislice


def subslice_spec(atoms, lz, n_slices):
    spec_lz = lz / n_slices
    subslices = []
    _, _, _, atom_z, _, _, _, _ = zip(*atoms)
    atom_z = numpy.array(atom_z)
    for i in range(n_slices):
        z0 = i * spec_lz
        z1 = (i + 1) * spec_lz
        if i == n_slices - 1:
            z1 = max(z1, numpy.max(atom_z)) + 1
        selection = (atom_z >= z0) & (atom_z < z1)
        subslices.append([atoms[i] for i in numpy.arange(0, len(atoms))[selection]])
    return subslices, spec_lz


def multi_multislice(system_conf, input_multislice, i, n_slices, output_multislice):

    if i == 0:
        input_multislice.iw_type = "Plane_Wave"
    else:
        input_multislice.iw_type = "User_Defined"
        input_multislice.iw_psi = output_multislice.data.psi_coh
        input_multislice.iw_x = 0.5 * input_multislice.spec_lx
        input_multislice.iw_y = 0.5 * input_multislice.spec_ly

    input_multislice.obj_lens_zero_defocus_type = "Last"
    output_multislice = multem.wave_function(system_conf, input_multislice)

    return output_multislice


if __name__ == "__main__":

    # Create the system configuration
    system_conf = multem.SystemConfiguration()
    system_conf.precision = "float"
    system_conf.device = "device"

    # Create the input multislice configuration
    n_phonons = 50
    input_multislice = create_input_multislice(n_phonons, False)

    print("Standard")
    start_time = time.time()
    ewrs = multem.simulate(system_conf, input_multislice)
    print("Time taken: ", time.time() - start_time)

    print("Subslicing")
    start_time = time.time()

    # Create the input multislice configuration
    n_slices = 4

    subslices, input_multislice.spec_lz = subslice_spec(
        input_multislice.spec_atoms, input_multislice.spec_lz, n_slices
    )

    output_multislice = multem.simulate(system_conf, input_multislice, subslices)

    print("Time taken: ", time.time() - start_time)

    from matplotlib import pylab

    a = numpy.array(ewrs.data[-1].m2psi_tot)
    b = numpy.array(output_multislice.data[-1].m2psi_tot)

    import pickle

    with open("standard.pickle", "wb") as outfile:
        pickle.dump(a, outfile)
    with open("subsliced.pickle", "wb") as outfile:
        pickle.dump(b, outfile)
    print(numpy.max(numpy.abs(a - b)))
    # pylab.imshow(a)
    # pylab.show()
    pylab.imshow(b - a)
    pylab.show()

    # psi_tot = numpy.zeros(shape=(input_multislice.nx, input_multislice.ny))

    # for p in range(n_phonons):

    #   input_multislice.pn_nconf = p
    #   output_multislice = multem.Output()
    #   for i in range(n_slices):
    #       input_multislice.spec_atoms = subslices[i]
    #       output_multislice = multi_multislice(system_conf, input_multislice, i,
    #                                            n_slices, output_multislice)
    #       input_multislice.pn_seed = input_multislice.pn_seed + 1
    #   end = len(output_multislice.data)-1
    #   psi_tot =  psi_tot + numpy.abs(output_multislice.data[end].psi_coh)**2

# subslice_ewrs =  psi_tot/n_phonons
