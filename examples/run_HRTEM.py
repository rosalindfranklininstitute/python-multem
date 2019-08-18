import multem
import numpy
import pickle
from cu001_crystal import cu001_crystal

input_multislice = multem.Input()
system_conf = multem.SystemConfiguration()

system_conf.precision = "float"
system_conf.device = "host"

# Set simulation experiment
input_multislice.simulation_type = "HRTEM"

# Electron-Specimen interaction model
input_multislice.interaction_model = "Multislice"

# Potential slicing
input_multislice.potential_slicing = "Planes"

# Electron-Phonon interaction model
input_multislice.pn_model = "Frozen_Phonon"
input_multislice.pn_coh_contrib = 0
input_multislice.pn_single_conf = False
input_multislice.pn_nconf = 10
input_multislice.pn_dim = 110
input_multislice.pn_seed = 300183

# Specimen information
na = 16
nb = 16
nc = 20
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
) = cu001_crystal(na, nb, nc, ncu, rms3d)

# Specimen thickness
input_multislice.thick_type = "Through_Thick"
input_multislice.thick = numpy.arange(c, 1000, c)

# x-y sampling
input_multislice.nx = 1024
input_multislice.ny = 1024
input_multislice.bwl = False

# Microscope parameters
input_multislice.E_0 = 300
input_multislice.theta = 0.0
input_multislice.phi = 0.0

# Illumination model
input_multislice.illumination_model = "Partial_Coherent"
input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

# Condenser lens
# source spread function
ssf_sigma = multem.mrad_to_sigma(input_multislice.E_0, 0.02)
input_multislice.cond_lens_ssf_sigma = ssf_sigma
input_multislice.cond_lens_ssf_npoints = 4

# Objective lens
input_multislice.obj_lens_m = 0
input_multislice.obj_lens_c_10 = 20
input_multislice.obj_lens_c_30 = 0.04
input_multislice.obj_lens_c_50 = 0.00
input_multislice.obj_lens_c_12 = 0.0
input_multislice.obj_lens_phi_12 = 0.0
input_multislice.obj_lens_c_23 = 0.0
input_multislice.obj_lens_phi_23 = 0.0
input_multislice.obj_lens_inner_aper_ang = 0.0
input_multislice.obj_lens_outer_aper_ang = 0.0

# defocus spread function
dsf_sigma = multem.iehwgd_to_sigma(32)
input_multislice.obj_lens_dsf_sigma = dsf_sigma
input_multislice.obj_lens_dsf_npoints = 5

# zero defocus reference
input_multislice.obj_lens_zero_defocus_type = "First"
input_multislice.obj_lens_zero_defocus_plane = 0

# Do the simulation
output_multislice = multem.simulate(system_conf, input_multislice)

data = {
    "dx": output_multislice.dx,
    "dy": output_multislice.dy,
    "x": numpy.array(output_multislice.x, dtype=numpy.float64),
    "y": numpy.array(output_multislice.y, dtype=numpy.float64),
    "thick": numpy.array(output_multislice.thick, dtype=numpy.float64),
    "data": [
        {"m2psi_tot": numpy.array(d.m2psi_tot, numpy.float64)}
        for d in output_multislice.data
    ],
}

pickle.dump(data, open("simulated_HRTEM.p", "wb"))
