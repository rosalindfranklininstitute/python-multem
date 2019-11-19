import multem
import numpy
import pickle
import time
from SrTiO3001Crystal import SrTiO3001_crystal

# from matplotlib import pylab


def create_input_multislice(n_phonons, single_phonon_conf=False):

    input_multislice = multem.Input()

    input_multislice.simulation_type = "HRTEM"
    input_multislice.interaction_model = "Multislice"
    input_multislice.potential_slicing = "Planes"

    input_multislice.potential_type = "Lobato_0_12"

    input_multislice.pn_model = "Still_Atom"
    # input_multislice.pn_model = "Frozen_Phonon"
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
    input_multislice.obj_lens_zero_defocus_type = "Last"
    input_multislice.obj_lens_zero_defocus_plane = 0

    input_multislice.output_area_ix_0 = 1
    input_multislice.output_area_iy_0 = 1
    input_multislice.output_area_ix_e = 1
    input_multislice.output_area_iy_e = 1

    return input_multislice


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

    subslices = list(
        multem.slice_spec_atoms(
            input_multislice.spec_atoms, input_multislice.spec_lz, n_slices
        )
    )
    output_multislice = multem.simulate(system_conf, input_multislice, subslices)

    print("Time taken: ", time.time() - start_time)

    a = numpy.array(ewrs.data[-1].m2psi_tot)
    if len(a) == 0:
        a = numpy.abs(numpy.array(ewrs.data[-1].psi_coh)) ** 2
    b = numpy.array(output_multislice.data[-1].m2psi_tot)

    with open("standard.pickle", "wb") as outfile:
        pickle.dump(a, outfile)
    with open("subsliced.pickle", "wb") as outfile:
        pickle.dump(b, outfile)
    print(numpy.max(numpy.abs(a - b)))
