import multem
import numpy
import pickle
import time
from SrTiO3001Crystal import SrTiO3001_crystal

# from matplotlib import pylab


def create_input_multislice(n_phonons, single_phonon_conf=False):

    # Initialise the input and system configuration
    input_multislice = multem.Input()

    # Set simulation experiment
    input_multislice.simulation_type = "EWRS"

    # Electron-Specimen interaction model
    input_multislice.interaction_model = "Multislice"
    input_multislice.potential_type = "Lobato_0_12"

    # Potential slicing
    input_multislice.potential_slicing = "dz_Proj"

    # Electron-Phonon interaction model
    input_multislice.pn_model = "Still_Atom"
    input_multislice.pn_coh_contrib = 0
    input_multislice.pn_single_conf = False
    input_multislice.pn_nconf = 10
    input_multislice.pn_dim = 110
    input_multislice.pn_seed = 300_183

    # Specimen thickness
    input_multislice.thick_type = "Whole_Spec"

    # Illumination model
    input_multislice.illumination_model = "Partial_Coherent"
    input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

    input_multislice.nx = 1024
    input_multislice.ny = 1024

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

    return input_multislice


if __name__ == "__main__":

    # Create the system configuration
    system_conf = multem.SystemConfiguration()
    system_conf.precision = "float"
    system_conf.device = "device"

    # Create the input multislice configuration
    n_phonons = 50
    input_multislice = create_input_multislice(n_phonons, False)

    # Create the specimen atoms
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

    print("Standard")
    start_time = time.time()
    ewrs = multem.simulate(system_conf, input_multislice)
    print("Time taken: ", time.time() - start_time)

    print("Subslicing")
    start_time = time.time()

    # Create the input multislice configuration
    n_slices = 4

    subslices = iter(list(
        multem.slice_spec_atoms(
            input_multislice.spec_atoms, input_multislice.spec_lz, n_slices
        )
    ))
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
