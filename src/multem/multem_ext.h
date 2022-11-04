/*
 *  multem_ext.h
 *
 *  Copyright (C) 2019 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the GPLv3 license, a copy of 
 *  which is included in the root directory of this package.
 */

#ifndef MULTEM_EXT_H
#define MULTEM_EXT_H

#include <array>
#include <cassert>
#include <complex>
#include <string>
#include <vector>
#include <multem/error.h>
#include <multem/masker.h>

namespace multem {

  /**
   * A class to represent atom data
   */
  class Atom {
  public:
      int element;
      double x;
      double y;
      double z;
      double sigma;
      double occupancy;
      int region;
      int charge;
      
      Atom()
        : element(0),
          x(0),
          y(0),
          z(0),
          sigma(0),
          occupancy(0),
          region(0),
          charge(0) {}

      Atom(
            int element_, 
            double x_, 
            double y_, 
            double z_, 
            double sigma_, 
            double occupancy_, 
            int region_, 
            int charge_)
        : element(element_),
          x(x_),
          y(y_),
          z(z_),
          sigma(sigma_),
          occupancy(occupancy_),
          region(region_),
          charge(charge_) {}
  };

  /**
   * A class to hold crystal parameters
   */
  class CrystalParameters {
  public:
    
    typedef std::vector<Atom> Layer;

    int na;
    int nb;
    int nc;
    double a;
    double b;
    double c;
    std::vector<Layer> layers;

    CrystalParameters()
      : na(0),
        nb(0),
        nc(0),
        a(0),
        b(0),
        c(0) {}

  };

  /**
   * A class to hold amorphous layer information
   */
  class AmorphousLayer {
  public:

    double z_0;
    double z_e;
    double dz;

    AmorphousLayer()
      : z_0(0),
        z_e(0),
        dz(0) {}

  };

  /**
   * A class to hold STEM detector information
   */
  class STEMDetector {
  public:

    class Angles {
    public:
      double inner_ang;
      double outer_ang;
      Angles()
        : inner_ang(0),
          outer_ang(0) {}
    };

    class Radial {
    public:
      double x;
      std::vector<double> fx;
      Radial()
        : x(0) {}
    };

    class Matrix {
    public:
      double R;
      std::vector<double> fR;
      Matrix()
        : R(0) {}
    };
  
    std::string type;
    std::vector<Angles> cir;
    std::vector<Radial> radial;
    std::vector<Matrix> matrix;
  };

  /**
   * A class to hold the input specification for the simulation
   */
  class Input {
  public:

    // Electron-specimen interaction model 
    std::string interaction_model; 
    std::string potential_type;  
    std::string operation_mode;
    std::size_t memory_size;
    bool reverse_multislice;
    
    // Electron-Phonon interaction model
    std::string pn_model;
    bool pn_coh_contrib;
    bool pn_single_conf;
    int pn_nconf;
    int pn_dim;
    int pn_seed;
    double static_B_factor;

    // Specimen information
    std::vector<Atom> spec_atoms;
    double spec_dz;
    double spec_lx;
    double spec_ly;
    double spec_lz;
    int spec_cryst_na;
    int spec_cryst_nb;
    int spec_cryst_nc;
    double spec_cryst_a;
    double spec_cryst_b;
    double spec_cryst_c;
    double spec_cryst_x0;
    double spec_cryst_y0;
    std::vector<AmorphousLayer> spec_amorp;

    // Specimen rotation
    double spec_rot_theta;
    std::array<double, 3> spec_rot_u0;
    std::string spec_rot_center_type;
    std::array<double, 3> spec_rot_center_p;

    // Specimen thickness
    std::string thick_type;
    std::vector<double> thick;

    // Potential slicing
    std::string potential_slicing;

    // X-Y sampling
    int nx;
    int ny;
    bool bwl;

    // Simulation type
    std::string simulation_type;

    // Incident wave
    std::string iw_type;
    std::vector< std::complex<double> > iw_psi;
    std::vector<double> iw_x;
    std::vector<double> iw_y;

    // Microscope parameters
    double E_0;
    double theta;
    double phi;

    // Illumination model
    std::string illumination_model;
    std::string temporal_spatial_incoh;

    // Condenser lens
    int cond_lens_m;
    double cond_lens_c_10;
    double cond_lens_c_12;
    double cond_lens_phi_12;
    double cond_lens_c_21;
    double cond_lens_phi_21;
    double cond_lens_c_23;
    double cond_lens_phi_23;
    double cond_lens_c_30;
    double cond_lens_c_32;
    double cond_lens_phi_32;
    double cond_lens_c_34;
    double cond_lens_phi_34;
    double cond_lens_c_41;
    double cond_lens_phi_41;
    double cond_lens_c_43;
    double cond_lens_phi_43;
    double cond_lens_c_45;
    double cond_lens_phi_45;
    double cond_lens_c_50;
    double cond_lens_c_52;
    double cond_lens_phi_52;
    double cond_lens_c_54;
    double cond_lens_phi_54;
    double cond_lens_c_56;
    double cond_lens_phi_56;
    double cond_lens_inner_aper_ang;
    double cond_lens_outer_aper_ang;

    // Source spread function
    double cond_lens_si_sigma;
    double cond_lens_si_a;
    double cond_lens_si_beta;
    int cond_lens_si_rad_npts;
    int cond_lens_si_azm_npts;
    
    // Defocus spread function
    double cond_lens_ti_a;
    double cond_lens_ti_sigma;
    double cond_lens_ti_beta;
    int cond_lens_ti_npts;

    // Zero defocus reference
    std::string cond_lens_zero_defocus_type;
    double cond_lens_zero_defocus_plane;

    // Objective lens
    int obj_lens_m;
    double obj_lens_c_10;
    double obj_lens_c_12;
    double obj_lens_phi_12;
    double obj_lens_c_21;
    double obj_lens_phi_21;
    double obj_lens_c_23;
    double obj_lens_phi_23;
    double obj_lens_c_30;
    double obj_lens_c_32;
    double obj_lens_phi_32;
    double obj_lens_c_34;
    double obj_lens_phi_34;
    double obj_lens_c_41;
    double obj_lens_phi_41;
    double obj_lens_c_43;
    double obj_lens_phi_43;
    double obj_lens_c_45;
    double obj_lens_phi_45;
    double obj_lens_c_50;
    double obj_lens_c_52;
    double obj_lens_phi_52;
    double obj_lens_c_54;
    double obj_lens_phi_54;
    double obj_lens_c_56;
    double obj_lens_phi_56;
    double obj_lens_inner_aper_ang;
    double obj_lens_outer_aper_ang;

    // Defocus spread function
    double obj_lens_ti_a;
    double obj_lens_ti_sigma;
    double obj_lens_ti_beta;
    int obj_lens_ti_npts;

    // The phase shift
    double phase_shift;

    // Zero defocus reference
    std::string obj_lens_zero_defocus_type;
    double obj_lens_zero_defocus_plane;

    // STEM detector
    STEMDetector detector;

    // Scanning area for ISTEM/STEM/EELS
    std::string scanning_type;
    bool scanning_square_pxs;
    bool scanning_periodic;
    int scanning_ns;
    double scanning_x0;
    double scanning_y0;
    double scanning_xe;
    double scanning_ye;

    // PED
    double ped_nrot;
    double ped_theta;

    // HCI
    double hci_nrot;
    double hci_theta;

    // EELS
    int eels_Z;
    double eels_E_loss;
    double eels_collection_angle;
    int eels_m_selection;
    std::string eels_channelling_type;

    // EFTEM
    int eftem_Z;
    double eftem_E_loss;
    double eftem_collection_angle;
    int eftem_m_selection;
    std::string eftem_channelling_type;

    // Output region
    int output_area_ix_0;
    int output_area_iy_0;
    int output_area_ix_e;
    int output_area_iy_e;

    /**
     * Set the default parameters
     */
    Input()
      : interaction_model("Multislice"), 
        potential_type("Lobato_0_12"),
        operation_mode("Normal"),
        memory_size(0),
        reverse_multislice(false),
        pn_model("Still_Atom"),
        pn_coh_contrib(false),
        pn_single_conf(false),
        pn_nconf(1),
        pn_dim(110),
        pn_seed(300183),
        static_B_factor(0),
        spec_dz(0.25),
        spec_lx(10),
        spec_ly(10),
        spec_lz(10),
        spec_cryst_na(1),
        spec_cryst_nb(1),
        spec_cryst_nc(1),
        spec_cryst_a(0),
        spec_cryst_b(0),
        spec_cryst_c(0),
        spec_cryst_x0(0),
        spec_cryst_y0(0),
        spec_rot_theta(0),
        spec_rot_u0({0, 0, 1}),
        spec_rot_center_type("geometric_center"),
        spec_rot_center_p({0, 0, 0}),
        thick_type("Whole_Spec"),
        thick(0),
        potential_slicing("Planes"),
        nx(256),
        ny(256),
        bwl(false),
        simulation_type("EWRS"),
        iw_type("Auto"),
        iw_psi(0),
        iw_x(0.0),
        iw_y(0.0),
        E_0(300.0),
        theta(0.0),
        phi(0.0),
        illumination_model("Partial_Coherent"),
        temporal_spatial_incoh("Temporal_Spatial"),
        cond_lens_m(0),
        cond_lens_c_10(14.0312),
        cond_lens_c_12(0.0),
        cond_lens_phi_12(0.0),
        cond_lens_c_21(0.0),
        cond_lens_phi_21(0.0),
        cond_lens_c_23(0.0),
        cond_lens_phi_23(0.0),
        cond_lens_c_30(1e-3),
        cond_lens_c_32(0.0),
        cond_lens_phi_32(0.0),
        cond_lens_c_34(0.0),
        cond_lens_phi_34(0.0),
        cond_lens_c_41(0.0),
        cond_lens_phi_41(0.0),
        cond_lens_c_43(0.0),
        cond_lens_phi_43(0.0),
        cond_lens_c_45(0.0),
        cond_lens_phi_45(0.0),
        cond_lens_c_50(0.0),
        cond_lens_c_52(0.0),
        cond_lens_phi_52(0.0),
        cond_lens_c_54(0.0),
        cond_lens_phi_54(0.0),
        cond_lens_c_56(0.0),
        cond_lens_phi_56(0.0),
        cond_lens_inner_aper_ang(0.0),
        cond_lens_outer_aper_ang(21.0),
        cond_lens_si_sigma(0.0072),
        cond_lens_si_a(1.0),
        cond_lens_si_beta(0.0),
        cond_lens_si_rad_npts(4),
        cond_lens_si_azm_npts(4),
        cond_lens_ti_a(1.0),
        cond_lens_ti_sigma(32),
        cond_lens_ti_beta(0.0),
        cond_lens_ti_npts(10),
        cond_lens_zero_defocus_type("First"),
        cond_lens_zero_defocus_plane(0),
        obj_lens_m(0),
        obj_lens_c_10(14.0312),
        obj_lens_c_12(0.0),
        obj_lens_phi_12(0.0),
        obj_lens_c_21(0.0),
        obj_lens_phi_21(0.0),
        obj_lens_c_23(0.0),
        obj_lens_phi_23(0.0),
        obj_lens_c_30(1e-3),
        obj_lens_c_32(0.0),
        obj_lens_phi_32(0.0),
        obj_lens_c_34(0.0),
        obj_lens_phi_34(0.0),
        obj_lens_c_41(0.0),
        obj_lens_phi_41(0.0),
        obj_lens_c_43(0.0),
        obj_lens_phi_43(0.0),
        obj_lens_c_45(0.0),
        obj_lens_phi_45(0.0),
        obj_lens_c_50(0.0),
        obj_lens_c_52(0.0),
        obj_lens_phi_52(0.0),
        obj_lens_c_54(0.0),
        obj_lens_phi_54(0.0),
        obj_lens_c_56(0.0),
        obj_lens_phi_56(0.0),
        obj_lens_inner_aper_ang(0.0),
        obj_lens_outer_aper_ang(24.0),
        obj_lens_ti_a(1.0),
        obj_lens_ti_sigma(32),
        obj_lens_ti_beta(0.0),
        obj_lens_ti_npts(10),
        obj_lens_zero_defocus_type("First"),
        obj_lens_zero_defocus_plane(0),
        phase_shift(0),
        scanning_type("Line"),
        scanning_square_pxs(false),
        scanning_periodic(true),
        scanning_ns(10),
        scanning_x0(0.0),
        scanning_y0(0.0),
        scanning_xe(4.078),
        scanning_ye(4.078),
        ped_nrot(360),
        ped_theta(3.0),
        hci_nrot(360),
        hci_theta(3.0),
        eels_Z(79),
        eels_E_loss(80),
        eels_collection_angle(100),
        eels_m_selection(3),
        eels_channelling_type("Single_Channelling"),
        eftem_Z(79),
        eftem_E_loss(80),
        eftem_collection_angle(100),
        eftem_m_selection(3),
        eftem_channelling_type("Single_Channelling"),
        output_area_ix_0(1),
        output_area_iy_0(1),
        output_area_ix_e(1),
        output_area_iy_e(1) {}
  };

  /**
   * A class to hold the system configuration
   */
  class SystemConfiguration {
  public:

    std::string device;
    std::string precision;
    std::size_t cpu_ncores;
    std::size_t cpu_nthread;
    std::size_t gpu_device;
    std::size_t gpu_nstream;
    
    SystemConfiguration()
      : device("device"),
        precision("float"),
        cpu_ncores(1),
        cpu_nthread(1),
        gpu_device(0),
        gpu_nstream(1) {}

  };

  /**
   * A class to represent an image
   */
  template <typename T>
  class Image {
  public:

    typedef T value_type;
    typedef std::array<std::size_t, 2> shape_type;

    std::vector<value_type> data;
    shape_type shape;

    Image()
      : shape({ 0, 0 }) {}

    /**
     * Construct the image from the pointer
     * @param data_ The data pointer
     * @param shape_ The image shape (Y, X)
     */
    template <typename U>
    Image(const U *data_, shape_type shape_)
      : shape(shape_) {
      std::size_t size = shape[0]*shape[1];
      data.assign(data_, data_ + size);
    }
  };

  /**
   * A class to represent a complex image
   */
  template <typename T>
  class Image< std::complex<T> > {
  public:
    
    typedef std::complex<T> value_type;
    typedef std::array<std::size_t, 2> shape_type;

    std::vector<value_type> data;
    shape_type shape;

    Image()
      : shape({ 0, 0 }) {}

    /**
     * Construct the image from the pointer
     * @param data_ The data pointer
     * @param shape_ The image shape (Y, X)
     */
    template <typename U>
    Image(const U *data_, shape_type shape_)
      : shape(shape_) {
      std::size_t size = shape[0]*shape[1];
      data.resize(size);
      for (auto i = 0; i < size; ++i) {
        data[i] = value_type(data_[i].real(), data_[i].imag());
      }
    }
    
  };

  /**
   * A class to hold output data
   */
  class Data {
  public:

    std::vector< Image<double> > image_tot;
    std::vector< Image<double> > image_coh;
    
    Image<double> m2psi_tot;
    Image<double> m2psi_coh;
    Image<double> V;
    Image< std::complex<double> > psi_coh;

  };

  /**
   * A class to contain output from the simulation
   */
  class Output {
  public:
      
    double dx;
    double dy;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> thick;
    std::vector<Data> data; 

  };

  /**
   * Run the simulation
   * @param config The system configuration
   * @param input The input
   * @returns The simulation results
   */
  Output simulate(SystemConfiguration config, Input input);
  
  /**
   * Run the simulation
   * @param config The system configuration
   * @param input The input
   * @returns The simulation results
   */
  Output simulate_with_ice_approximation(
      SystemConfiguration config, 
      Input input, 
      const Masker &masker);
 
  /**
   * Run the simulation. This overload performs the simulation for a sample
   * that is split into a number of slices. This might be done because the
   * sample is too big to simulate all at once for example. This function
   * was written based on a script from Thomas Friedrich.
   * @param config The system configuration
   * @param input The input
   * @param first The first slice
   * @param slice The last slice
   * @returns The simulation results
   */
  template <typename SliceIterator>
  Output simulate_slices(
      SystemConfiguration config, 
      Input input, 
      SliceIterator first, 
      SliceIterator last) {

    // The result
    Output result;

    // Get the number of phonons
    std::size_t n_phonons = (
      input.pn_model == "Frozen_Phonon"
        ? (input.pn_single_conf ? 1 : input.pn_nconf)
        : 1);
    MULTEM_ASSERT(n_phonons > 0);
    MULTEM_ASSERT(input.simulation_type == "EWRS");

    // Save the simulation type and set it to exit wave reconstruction
    auto simulation_type = input.simulation_type;

    // Check some input. This function requires that the input wave type is a
    // plane wave  
    MULTEM_ASSERT(input.iw_type == "Plane_Wave" || input.iw_type == "Auto");

    // Need to override some stuff. This is from a script from Thomas Friedrich
    input.pn_coh_contrib = true;
    input.pn_single_conf = true;
    
    // Need to set the zero defocus type to Last because in the multi slice
    // calculation for the exit wave the only position that makes physical sense
    // is the last plane at the exit of the sameple. Typically, we always want
    // this to be the last position in the forward simulation.
    input.obj_lens_zero_defocus_type = "Last";

    // The total wave vector
    std::size_t image_size = input.nx * input.ny;
    std::vector<double> m2psi_tot(image_size);

    // Loop through the number of phonons
    for (std::size_t p = 0; p < n_phonons; ++p) {

      // Set the current phonon
      input.pn_nconf = p+1;

      // Loop through the slices of the sample
      std::size_t count_non_zero = 0;
      for (auto slice = first; slice != last; ++slice) {
        //
        // Get the specimen size and set the input atoms
        // It's convenient to assign the s0 and lz to local variables
        // to avoid dereferencing the iterator many times in the loop below
        // which in the python wrapper calls the GIL and will slow things down.
        auto item = *slice;
        double spec_z0 = item.spec_z0;
        double spec_lz = item.spec_lz;
        MULTEM_ASSERT(spec_z0 >= 0);
        MULTEM_ASSERT(spec_lz > 0);
        input.spec_lz = spec_lz;
        input.spec_atoms = item.spec_atoms;
        if (input.spec_atoms.size() == 0) {
          continue;
        }

        // We need to shift the Z coordinates of the atoms such that they lie
        // within the box
        for (auto &atom : input.spec_atoms) {
          MULTEM_ASSERT(atom.z >= spec_z0);
          atom.z -= spec_z0;
          MULTEM_ASSERT(atom.z < spec_lz);
        }

        // On the first iteration then set the input wave as a plane wave and on
        // the subsequent iterations set the input wave as the previous exit
        // wave.
        if (count_non_zero == 0) {
          input.iw_type = "Plane_Wave";
        } else {
          input.iw_type = "User_Define_Wave";
          MULTEM_ASSERT(result.data.size() == 1);
          input.iw_psi = result.data.back().psi_coh.data;
          input.iw_x = { 0.5*input.spec_lx };
          input.iw_y = { 0.5*input.spec_ly };
          MULTEM_ASSERT(result.data.back().psi_coh.data.size() == image_size);
        }

        // Run the simulation
        result = simulate(config, input);

        // Check the output
        MULTEM_ASSERT(result.data.size() != 0);

        // Increment the random seed
        count_non_zero++;
        input.pn_seed++;
      }

      // Check some slices had atoms
      MULTEM_ASSERT(count_non_zero > 0);

      // Set the contribution to the total 
      MULTEM_ASSERT(result.data.size() == 1);
      if (result.data.back().m2psi_tot.data.size() > 0) {
        MULTEM_ASSERT(m2psi_tot.size() == result.data.back().m2psi_tot.data.size());
        for (std::size_t i = 0; i < m2psi_tot.size(); ++i) {
          m2psi_tot[i] += result.data.back().m2psi_tot.data[i];
        }
      } else {
        MULTEM_ASSERT(m2psi_tot.size() == result.data.back().psi_coh.data.size());
        for (std::size_t i = 0; i < m2psi_tot.size(); ++i) {
          m2psi_tot[i] += std::pow(std::abs(result.data.back().psi_coh.data[i]), 2);
        }
      }
    }

    // Divide by the number of phonons
    for (auto &x : m2psi_tot) {
      x /= n_phonons;
    }

    // Get the image shape
    auto image_shape = (
        result.data.back().m2psi_tot.data.size() > 0 
          ? result.data.back().m2psi_tot.shape
          : result.data.back().psi_coh.shape);

    // Set the result
    result.data.back().m2psi_tot = Image<double>(
        m2psi_tot.data(), 
        image_shape);

    // Return the result
    return result;
  }

  /* Function callback for projected potential */
  typedef std::function<void(double, double, Image<double>)> projected_potential_callback;

  /**
   * Run the simulation
   * @param config The system configuration
   * @param input The input
   * @returns The simulation results
   */
  Output compute_projected_potential(
      SystemConfiguration config, 
      Input input, 
      projected_potential_callback callback);
  
  /**
   * Run the simulation
   * @param config The system configuration
   * @param input The input
   * @returns The simulation results
   */
  Output compute_projected_potential_with_ice_approximation(
      SystemConfiguration config, 
      Input input, 
      const Masker &masker,
      projected_potential_callback callback);

  /**
   * @returns True/False if the GPU is available
   */
  bool is_gpu_available();

  /**
   * @returns The number of available GPUs
   */
  int number_of_gpu_available();
  
  /**
   * Compute the CTF
   */
  Image< std::complex<double> > compute_ctf( 
      SystemConfiguration config, 
      Input input);

  double mrad_to_sigma(double E0, double theta);
  double iehwgd_to_sigma(double value);
  double hwhm_to_sigma(double value);

  std::vector<Atom> crystal_by_layers(const CrystalParameters &params);
  std::vector< std::pair<double, double> > compute_V_params(
      std::string potential_type, 
      std::size_t Z, 
      int charge);


  /**
   * Tests
   */
  void test_ice_potential_approximation();
	void test_masker();
}

#endif
