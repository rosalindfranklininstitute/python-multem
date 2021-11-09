import multem
import numpy
import pickle
import time
from Si001Crystal import Si001_crystal

st = time.time()
print("GPU available: %s" % multem.is_gpu_available())

input_multislice = multem.Input()
system_conf = multem.SystemConfiguration()

system_conf.precision = "float"
system_conf.device = "device"

# Set simulation experiment
input_multislice.simulation_type = "CBED"

# Electron-Specimen interaction model
input_multislice.interaction_model = "Multislice"
input_multislice.potential_type = "Lobato_0_12"

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
na = 8
nb = 8
nc = 40
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
) = Si001_crystal(na, nb, nc, ncu, rms3d)

# Specimen thickness
input_multislice.thick_type = "Whole_Spec"
input_multislice.thick = []

# x-y sampling
input_multislice.nx = 1024
input_multislice.ny = 1024
input_multislice.bwl = False

# Microscope parameters
input_multislice.E_0 = 100
input_multislice.theta = 0.0
input_multislice.phi = 0.0

# Illumination model
input_multislice.illumination_model = "Full_Integration"
input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

# Set the incident wave
input_multislice.iw_type = "Auto"
# input_multislice.iw_psi = read_psi_0_multem(input_multislice.nx, input_multislice.ny)
input_multislice.iw_x = [0]  # input_multislice.spec_lx/2
input_multislice.iw_y = [0]  # input_multislice.spec_ly/2

# Condenser lens
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

# defocus spread function
ti_sigma = multem.iehwgd_to_sigma(32)
input_multislice.cond_lens_ti_a = 1.0
input_multislice.cond_lens_ti_sigma = ti_sigma
input_multislice.cond_lens_ti_beta = 0.0
input_multislice.cond_lens_ti_npts = 4

# Source spread function
si_sigma = multem.hwhm_to_sigma(0.45)
input_multislice.cond_lens_si_a = 1.0
input_multislice.cond_lens_si_sigma = si_sigma
input_multislice.cond_lens_si_beta = 0.0
input_multislice.cond_lens_si_rad_npts = 4
input_multislice.cond_lens_si_azm_npts = 4

# Zero defocus reference
input_multislice.cond_lens_zero_defocus_type = "User_Define"
input_multislice.cond_lens_zero_defocus_plane = 0

# Do the simulation
output_multislice = multem.simulate(system_conf, input_multislice)
print("Time: %.2f" % (time.time() - st))

data = []
for i in range(len(output_multislice.data)):
    m2psi_tot = output_multislice.data[i].m2psi_tot
    data.append(numpy.array(m2psi_tot))

pickle.dump(data, open("example_CBED.p", "wb"), protocol=2)
