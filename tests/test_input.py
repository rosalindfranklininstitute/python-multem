import pytest
import multem
import numpy
import pickle


def test_input():
    stem_detector = multem.STEMDetector()
    stem_detector.type = "Test"
    stem_detector.cir = [(0, 1), (2, 3)]
    stem_detector.radial = [(0, [1, 2, 3, 4]), (2, [5, 6, 8, 9])]
    stem_detector.matrix = [(3, [1, 2, 3, 4]), (4, [5, 6, 7, 8])]

    input = multem.Input()
    input.interaction_model = "Interaction Model"
    input.potential_type = "Potential Type"
    input.operation_mode = "Operation Mode"
    input.memory_size = 10
    input.reverse_multislice = False

    input.pn_model = "Phonon Interaction Model"
    input.pn_coh_contrib = True
    input.pn_single_conf = False
    input.pn_nconf = 20
    input.pn_dim = 30
    input.pn_seed = 40

    input.spec_atoms = multem.AtomList(
        [(1, 2, 3, 4, 5, 6, 7, 8), (2, 3, 4, 5, 6, 7, 8, 9)]
    )

    input.spec_dz = 50.1
    input.spec_lx = 60.1
    input.spec_ly = 70.1
    input.spec_lz = 80.1
    input.spec_cryst_na = 90
    input.spec_cryst_nb = 100
    input.spec_cryst_nc = 110
    input.spec_cryst_a = 120.1
    input.spec_cryst_b = 130.1
    input.spec_cryst_c = 140.1
    input.spec_cryst_x0 = 150.1
    input.spec_cryst_y0 = 160.1
    input.spec_amorp = [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]

    input.spec_rot_theta = 0
    input.spec_rot_u0 = (1, 2, 3)
    input.spec_rot_center_type = "Spec Rot Center Type"
    input.spec_rot_center_p = (4, 5, 6)

    input.thick_type = "Thick Type"
    input.thick = [1.1, 2.2, 3.3, 4.4]

    input.potential_slicing = "Potential Slicing"

    input.nx = 1
    input.ny = 2
    input.bwl = True

    input.simulation_type = "Simulation Type"

    input.iw_type = "IW Type"
    input.iw_psi = [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]
    input.iw_x = [1.1, 2.2, 3.3, 4.4, 5.5]
    input.iw_y = [6.6, 7.7, 8.8, 9.9, 0.0]

    input.E_0 = 3.1
    input.theta = 4.1
    input.phi = 5.9

    input.illumination_model = "Illumination Model"
    input.temporal_spatial_incoh = "Temporal Spatial Incoherence"

    input.cond_lens_m = 1
    input.cond_lens_c_10 = 0.1
    input.cond_lens_c_12 = 0.2
    input.cond_lens_phi_12 = 0.3
    input.cond_lens_c_21 = 0.4
    input.cond_lens_phi_21 = 0.5
    input.cond_lens_c_23 = 0.6
    input.cond_lens_phi_23 = 0.7
    input.cond_lens_c_30 = 0.8
    input.cond_lens_c_32 = 0.9
    input.cond_lens_phi_32 = 1.0
    input.cond_lens_c_34 = 1.1
    input.cond_lens_phi_34 = 1.2
    input.cond_lens_c_41 = 1.3
    input.cond_lens_phi_41 = 1.4
    input.cond_lens_c_43 = 1.5
    input.cond_lens_phi_43 = 1.6
    input.cond_lens_c_45 = 1.7
    input.cond_lens_phi_45 = 1.8
    input.cond_lens_c_50 = 1.9
    input.cond_lens_c_52 = 2.0
    input.cond_lens_phi_52 = 2.1
    input.cond_lens_c_54 = 2.2
    input.cond_lens_phi_54 = 2.3
    input.cond_lens_c_56 = 2.4
    input.cond_lens_phi_56 = 2.5
    input.cond_lens_inner_aper_ang = 2.6
    input.cond_lens_outer_aper_ang = 2.7

    input.cond_lens_si_a = 0.1
    input.cond_lens_si_sigma = 0.2
    input.cond_lens_si_beta = 0.3
    input.cond_lens_si_rad_npts = 2
    input.cond_lens_si_azm_npts = 3

    input.cond_lens_ti_a = 0.1
    input.cond_lens_ti_sigma = 0.2
    input.cond_lens_ti_beta = 0.3
    input.cond_lens_ti_npts = 4

    input.cond_lens_zero_defocus_type = "Cond Lens Zero Defocus Type"
    input.cond_lens_zero_defocus_plane = 0.123

    input.obj_lens_m = 12
    input.obj_lens_c_10 = 0.1
    input.obj_lens_c_12 = 0.2
    input.obj_lens_phi_12 = 0.3
    input.obj_lens_c_21 = 0.4
    input.obj_lens_phi_21 = 0.5
    input.obj_lens_c_23 = 0.6
    input.obj_lens_phi_23 = 0.7
    input.obj_lens_c_30 = 0.8
    input.obj_lens_c_32 = 0.9
    input.obj_lens_phi_32 = 0.10
    input.obj_lens_c_34 = 0.11
    input.obj_lens_phi_34 = 0.12
    input.obj_lens_c_41 = 0.13
    input.obj_lens_phi_41 = 0.14
    input.obj_lens_c_43 = 0.15
    input.obj_lens_phi_43 = 0.16
    input.obj_lens_c_45 = 0.17
    input.obj_lens_phi_45 = 0.18
    input.obj_lens_c_50 = 0.19
    input.obj_lens_c_52 = 0.20
    input.obj_lens_phi_52 = 0.21
    input.obj_lens_c_54 = 0.22
    input.obj_lens_phi_54 = 0.23
    input.obj_lens_c_56 = 0.24
    input.obj_lens_phi_56 = 0.25
    input.obj_lens_inner_aper_ang = 0.26
    input.obj_lens_outer_aper_ang = 0.27

    input.obj_lens_ti_sigma = 0.1
    input.obj_lens_ti_npts = 20

    input.obj_lens_zero_defocus_type = "Obj Lens Zero Defocus Type"
    input.obj_lens_zero_defocus_plane = 1.1

    input.detector = stem_detector

    input.scanning_type = "Scanning Type"
    input.scanning_periodic = True
    input.scanning_ns = 20
    input.scanning_x0 = 0.1
    input.scanning_y0 = 0.2
    input.scanning_xe = 0.3
    input.scanning_ye = 0.4

    input.ped_nrot = 0.5
    input.ped_theta = 0.6

    input.hci_nrot = 0.7
    input.hci_theta = 0.8

    input.eels_Z = 20
    input.eels_E_loss = 0.9
    input.eels_collection_angle = 10.1
    input.eels_m_selection = 30
    input.eels_channelling_type = "EELS Channelling Type"

    input.eftem_Z = 50
    input.eftem_E_loss = 0.1
    input.eftem_collection_angle = 0.2
    input.eftem_m_selection = 60
    input.eftem_channelling_type = "EFTEM Channelling Type"

    input.output_area_ix_0 = 10
    input.output_area_iy_0 = 20
    input.output_area_ix_e = 30
    input.output_area_iy_e = 40

    def check():
        assert input.interaction_model == "Interaction Model"
        assert input.potential_type == "Potential Type"
        assert input.operation_mode == "Operation Mode"
        assert input.memory_size == 10
        assert input.reverse_multislice == False

        assert input.pn_model == "Phonon Interaction Model"
        assert input.pn_coh_contrib == True
        assert input.pn_single_conf == False
        assert input.pn_nconf == 20
        assert input.pn_dim == 30
        assert input.pn_seed == 40

        assert input.spec_atoms == pytest.approx(
            numpy.array([(1, 2, 3, 4, 5, 6, 7, 8), (2, 3, 4, 5, 6, 7, 8, 9)])
        )

        assert input.spec_dz == 50.1
        assert input.spec_lx == 60.1
        assert input.spec_ly == 70.1
        assert input.spec_lz == 80.1
        assert input.spec_cryst_na == 90
        assert input.spec_cryst_nb == 100
        assert input.spec_cryst_nc == 110
        assert input.spec_cryst_a == 120.1
        assert input.spec_cryst_b == 130.1
        assert input.spec_cryst_c == 140.1
        assert input.spec_cryst_x0 == 150.1
        assert input.spec_cryst_y0 == 160.1
        assert input.spec_amorp == pytest.approx(
            numpy.array([(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)])
        )

        assert input.spec_rot_theta == 0
        assert input.spec_rot_u0 == pytest.approx((1, 2, 3))
        assert input.spec_rot_center_type == "Spec Rot Center Type"
        assert input.spec_rot_center_p == pytest.approx((4, 5, 6))

        assert input.thick_type == "Thick Type"
        assert tuple(input.thick) == pytest.approx([1.1, 2.2, 3.3, 4.4])

        assert input.potential_slicing == "Potential Slicing"

        assert input.nx == 1
        assert input.ny == 2
        assert input.bwl == True

        assert input.simulation_type == "Simulation Type"

        assert input.iw_type == "IW Type"
        assert input.iw_psi == [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]
        assert input.iw_x == [1.1, 2.2, 3.3, 4.4, 5.5]
        assert input.iw_y == [6.6, 7.7, 8.8, 9.9, 0.0]

        assert input.E_0 == 3.1
        assert input.theta == 4.1
        assert input.phi == 5.9

        assert input.illumination_model == "Illumination Model"
        assert input.temporal_spatial_incoh == "Temporal Spatial Incoherence"

        assert input.cond_lens_m == 1
        assert input.cond_lens_c_10 == 0.1
        assert input.cond_lens_c_12 == 0.2
        assert input.cond_lens_phi_12 == 0.3
        assert input.cond_lens_c_21 == 0.4
        assert input.cond_lens_phi_21 == 0.5
        assert input.cond_lens_c_23 == 0.6
        assert input.cond_lens_phi_23 == 0.7
        assert input.cond_lens_c_30 == 0.8
        assert input.cond_lens_c_32 == 0.9
        assert input.cond_lens_phi_32 == 1.0
        assert input.cond_lens_c_34 == 1.1
        assert input.cond_lens_phi_34 == 1.2
        assert input.cond_lens_c_41 == 1.3
        assert input.cond_lens_phi_41 == 1.4
        assert input.cond_lens_c_43 == 1.5
        assert input.cond_lens_phi_43 == 1.6
        assert input.cond_lens_c_45 == 1.7
        assert input.cond_lens_phi_45 == 1.8
        assert input.cond_lens_c_50 == 1.9
        assert input.cond_lens_c_52 == 2.0
        assert input.cond_lens_phi_52 == 2.1
        assert input.cond_lens_c_54 == 2.2
        assert input.cond_lens_phi_54 == 2.3
        assert input.cond_lens_c_56 == 2.4
        assert input.cond_lens_phi_56 == 2.5
        assert input.cond_lens_inner_aper_ang == 2.6
        assert input.cond_lens_outer_aper_ang == 2.7

        assert input.cond_lens_si_a == 0.1
        assert input.cond_lens_si_sigma == 0.2
        assert input.cond_lens_si_beta == 0.3
        assert input.cond_lens_si_rad_npts == 2
        assert input.cond_lens_si_azm_npts == 3

        assert input.cond_lens_ti_a == 0.1
        assert input.cond_lens_ti_sigma == 0.2
        assert input.cond_lens_ti_beta == 0.3
        assert input.cond_lens_ti_npts == 4

        assert input.cond_lens_zero_defocus_type == "Cond Lens Zero Defocus Type"
        assert input.cond_lens_zero_defocus_plane == 0.123

        assert input.obj_lens_m == 12
        assert input.obj_lens_c_10 == 0.1
        assert input.obj_lens_c_12 == 0.2
        assert input.obj_lens_phi_12 == 0.3
        assert input.obj_lens_c_21 == 0.4
        assert input.obj_lens_phi_21 == 0.5
        assert input.obj_lens_c_23 == 0.6
        assert input.obj_lens_phi_23 == 0.7
        assert input.obj_lens_c_30 == 0.8
        assert input.obj_lens_c_32 == 0.9
        assert input.obj_lens_phi_32 == 0.10
        assert input.obj_lens_c_34 == 0.11
        assert input.obj_lens_phi_34 == 0.12
        assert input.obj_lens_c_41 == 0.13
        assert input.obj_lens_phi_41 == 0.14
        assert input.obj_lens_c_43 == 0.15
        assert input.obj_lens_phi_43 == 0.16
        assert input.obj_lens_c_45 == 0.17
        assert input.obj_lens_phi_45 == 0.18
        assert input.obj_lens_c_50 == 0.19
        assert input.obj_lens_c_52 == 0.20
        assert input.obj_lens_phi_52 == 0.21
        assert input.obj_lens_c_54 == 0.22
        assert input.obj_lens_phi_54 == 0.23
        assert input.obj_lens_c_56 == 0.24
        assert input.obj_lens_phi_56 == 0.25
        assert input.obj_lens_inner_aper_ang == 0.26
        assert input.obj_lens_outer_aper_ang == 0.27

        assert input.obj_lens_ti_sigma == 0.1
        assert input.obj_lens_ti_npts == 20

        assert input.obj_lens_zero_defocus_type == "Obj Lens Zero Defocus Type"
        assert input.obj_lens_zero_defocus_plane == 1.1

        assert input.detector.type == "Test"
        assert input.detector.cir == [(0, 1), (2, 3)]
        assert input.detector.radial == [(0, [1, 2, 3, 4]), (2, [5, 6, 8, 9])]
        assert input.detector.matrix == [(3, [1, 2, 3, 4]), (4, [5, 6, 7, 8])]

        assert input.scanning_type == "Scanning Type"
        assert input.scanning_periodic == True
        assert input.scanning_ns == 20
        assert input.scanning_x0 == 0.1
        assert input.scanning_y0 == 0.2
        assert input.scanning_xe == 0.3
        assert input.scanning_ye == 0.4

        assert input.ped_nrot == 0.5
        assert input.ped_theta == 0.6

        assert input.hci_nrot == 0.7
        assert input.hci_theta == 0.8

        assert input.eels_Z == 20
        assert input.eels_E_loss == 0.9
        assert input.eels_collection_angle == 10.1
        assert input.eels_m_selection == 30
        assert input.eels_channelling_type == "EELS Channelling Type"

        assert input.eftem_Z == 50
        assert input.eftem_E_loss == 0.1
        assert input.eftem_collection_angle == 0.2
        assert input.eftem_m_selection == 60
        assert input.eftem_channelling_type == "EFTEM Channelling Type"

        assert input.output_area_ix_0 == 10
        assert input.output_area_iy_0 == 20
        assert input.output_area_ix_e == 30
        assert input.output_area_iy_e == 40

    check()

    input = pickle.loads(pickle.dumps(input))

    check()
