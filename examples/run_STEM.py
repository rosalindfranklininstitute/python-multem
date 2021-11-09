import multem
import numpy
import pickle
import time
from Au110Crystal import Au110_crystal

st = time.time()
print("GPU available: %s" % multem.is_gpu_available())

input_multislice = multem.Input()
system_conf = multem.SystemConfiguration()

system_conf.precision = "float"
system_conf.device = "device"

# Set simulation experiment
input_multislice.simulation_type = "STEM"

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
nc = 5
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
) = Au110_crystal(na, nb, nc, ncu, rms3d)

# Specimen thickness
input_multislice.thick_type = "Through_Thick"
input_multislice.thick = numpy.arange(c / 2.0, 1000, c)

# x-y sampling
input_multislice.nx = 512
input_multislice.ny = 512
input_multislice.bwl = False

# Microscope parameters
input_multislice.E_0 = 300
input_multislice.theta = 0.0
input_multislice.phi = 0.0

# Illumination model
input_multislice.illumination_model = "Coherent"
input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

# Set the incident wave
# input_multislice.iw_type = "Auto"
# input_multislice.iw_psi = read_psi_0_multem(input_multislice.nx, input_multislice.ny)
input_multislice.iw_x = [0]  # input_multislice.spec_lx/2
input_multislice.iw_y = [0]  # input_multislice.spec_ly/2

# Condenser lens
input_multislice.cond_lens_m = 0
input_multislice.cond_lens_c_10 = 14.0312
input_multislice.cond_lens_c_30 = 1e-3
input_multislice.cond_lens_c_50 = 0.00
input_multislice.cond_lens_c_12 = 0.0
input_multislice.cond_lens_phi_12 = 0.0
input_multislice.cond_lens_c_23 = 0.0
input_multislice.cond_lens_phi_23 = 0.0
input_multislice.cond_lens_inner_aper_ang = 0.0
input_multislice.cond_lens_outer_aper_ang = 21.0

# defocus spread function
ti_sigma = multem.iehwgd_to_sigma(32)
input_multislice.cond_lens_ti_sigma = ti_sigma
input_multislice.cond_lens_ti_npts = 5

# Source spread function
si_sigma = multem.hwhm_to_sigma(0.45)
input_multislice.cond_lens_si_sigma = si_sigma
input_multislice.cond_lens_si_rad_npts = 8
input_multislice.cond_lens_si_azm_npts = 12

# Zero defocus reference
input_multislice.cond_lens_zero_defocus_type = "First"
input_multislice.cond_lens_zero_defocus_plane = 0

# STEM
input_multislice.scanning_type = "Area"
input_multislice.scanning_periodic = True
input_multislice.scanning_square_pxs = False
input_multislice.scanning_ns = 20
input_multislice.scanning_x0 = 3 * a
input_multislice.scanning_y0 = 3 * b
input_multislice.scanning_xe = 4 * a
input_multislice.scanning_ye = 4 * b

# Detector
input_multislice.detector.type = "Circular"
input_multislice.detector.cir = [(40, 160), (80, 160)]

# Do the simulation
output_multislice = multem.simulate(system_conf, input_multislice)
print("Time: %.2f" % (time.time() - st))

data = []
for i in range(len(output_multislice.data)):
    d = []
    for j in range(len(input_multislice.detector.cir)):
        d.append(numpy.array(output_multislice.data[i].image_tot[j]))
    data.append(d)

pickle.dump(data, open("example_STEM.p", "wb"), protocol=2)
