import multem
import numpy as np
from matplotlib import pylab

system_conf = multem.SystemConfiguration()
system_conf.precision = "float"
system_conf.device = "host"

input_multem = multem.Input()
input_multem.E_0 = 300
input_multem.theta = 0.0
input_multem.phi = 0.0

input_multem.spec_lx = 20
input_multem.spec_ly = 20

input_multem.nx = 1024
input_multem.ny = 1024

# Init psi0
Y, X = np.mgrid[0 : input_multem.ny, 0 : input_multem.nx]
R = np.sqrt((Y / input_multem.ny - 1 / 2) ** 2 + (X / input_multem.nx - 1 / 2) ** 2)
A = np.exp(-0.5 * R**2)
psi0 = np.exp(1j * A)
psi0 = np.ones_like(psi0)
print(psi0.shape)
pylab.imshow(np.abs(psi0) ** 2)
pylab.show()

# Incident wave
input_multem.iw_type = "User_Define_Wave"
input_multem.iw_psi = psi0.flatten()
input_multem.iw_x = [0.5 * input_multem.spec_lx]
input_multem.iw_y = [0.5 * input_multem.spec_ly]

# Condenser lens
input_multem.cond_lens_m = 0
input_multem.cond_lens_c_10 = -150.00
input_multem.cond_lens_c_30 = 1e-03
input_multem.cond_lens_c_50 = 0.00
input_multem.cond_lens_c_12 = 0.0
input_multem.cond_lens_phi_12 = 0.0
input_multem.cond_lens_c_23 = 0.0
input_multem.cond_lens_phi_23 = 0.0
input_multem.cond_lens_inner_aper_ang = 0.0
input_multem.cond_lens_outer_aper_ang = 21.0
input_multem.cond_lens_ti_sigma = 32
input_multem.cond_lens_ti_npts = 10
input_multem.cond_lens_si_sigma = 0.2
input_multem.cond_lens_si_rad_npts = 8
input_multem.cond_lens_zero_defocus_type = "First"
input_multem.cond_lens_zero_defocus_plane = 0

input_multem.obj_lens_m = 0
input_multem.obj_lens_c_10 = 20
input_multem.obj_lens_c_30 = 0.04
input_multem.obj_lens_c_50 = 0.00
input_multem.obj_lens_c_12 = 0.0
input_multem.obj_lens_phi_12 = 0.0
input_multem.obj_lens_c_23 = 0.0
input_multem.obj_lens_phi_23 = 0.0
input_multem.obj_lens_inner_aper_ang = 0.0
input_multem.obj_lens_outer_aper_ang = 0.0

input_multem.simulation_type = "EWRS"
input_multem.potential_slicing = "dz_Proj"
input_multem.thick_type = "Whole_Spec"
input_multem.spec_atoms = multem.AtomList(
    [(8, 0.5 * input_multem.spec_lx, 0.5 * input_multem.spec_ly, 0, 0, 1, 0, 0)]
)
input_multem.interaction_model = "Multislice"
input_multem.pn_coh_contrib = 0
input_multem.pn_single_conf = False
input_multem.obj_lens_zero_defocus_type = "Last"

for x in np.arange(0.4, 0.6, 0.025) * input_multem.spec_lx:
    for y in np.arange(0.4, 0.6, 0.025) * input_multem.spec_ly:
        # input_multem.iw_x = [x];
        # input_multem.iw_y = [y];

        output_incident_wave = multem.simulate(system_conf, input_multem)
        image = np.array(output_incident_wave.data[0].psi_coh).T
        print(np.mean(np.abs(image)))
        pylab.imshow(np.abs(image) ** 2)
        pylab.show()
        # output_incident_wave = input_multem.ilc_incident_wave;
        # psi_0 = output_incident_wave.psi_0;
