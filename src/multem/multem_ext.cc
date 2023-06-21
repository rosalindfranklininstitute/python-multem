/*
 *  multem_ext.cc
 *
 *  Copyright (C) 2019 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the GPLv3 license, a copy of 
 *  which is included in the root directory of this package.
 */

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include <multem/multem_ext.h>

namespace py = pybind11;

// Make the vector of atoms opaque
PYBIND11_MAKE_OPAQUE(std::vector<multem::Atom>);

namespace pybind11 { namespace detail {

  /**
   * A wrapper around the mask compute funcion
   */
  py::array_t<int> Masker_compute(const multem::Masker &masker, double zs, double ze) {
    std::vector<int> mask(masker.image_size());
    masker.compute(zs, ze, mask.begin());
    return py::array_t<int>(mask.size(), mask.data());
  }

  /**
   * An iterator wrapper that extracts the pybind iterator into the desired 
   * C++ type. Importantly, the GIL is acquired whenever python code is called.
   */
  template <typename ValueType, typename Iterator>
  class SliceIterator {
  public:

    using value_type = ValueType;
    using reference = const ValueType&;
    using pointer = const ValueType*;

    SliceIterator(Iterator iterable)
      : iterable_(iterable) {}

    SliceIterator& operator++() {
      py::gil_scoped_acquire acquire;
      iterable_++;
      return *this;
    }

    SliceIterator operator++(int) {
      py::gil_scoped_acquire acquire;
      auto rv = *this;
      iterable_++;
      return rv;
    }
    
    reference operator*() {
      py::gil_scoped_acquire acquire;
      value_ = iterable_->template cast<value_type>();
      return value_;
    }

    pointer operator->() { 
      operator*(); 
      return &value_; 
    }

    friend
    bool operator==(const SliceIterator &a, const SliceIterator &b) { 
      py::gil_scoped_acquire acquire;
      return a.iterable_ == b.iterable_; 
    }

    friend
    bool operator!=(const SliceIterator &a, const SliceIterator &b) { 
      py::gil_scoped_acquire acquire;
      return a.iterable_ != b.iterable_;
    }

  private:
    Iterator iterable_;
    value_type value_;
  };

  /**
   * Make the slice iterator
   * @param iterator The pybind11 iterator to wrap
   * @returns The slice iterator
   */
  template <typename ValueType, typename Iterator>
  SliceIterator<ValueType, Iterator> make_slice_iterator(Iterator iterator) {
    return SliceIterator<ValueType, Iterator>(iterator);
  }

  /**
   * A class to hold the sub-sliced atoms
   */
  class Slice {
  public:
    std::vector<multem::Atom> spec_atoms;
    double spec_lz;
    double spec_z0;
    Slice()
      : spec_lz(0),
        spec_z0(0) {}
  };

  /**
   * Wrap the simulate function
   * @param config The system configuration
   * @param input The input
   * @param sequence The sequence
   * @returns The simulation results
   */
  multem::Output simulate_slices(
        multem::SystemConfiguration config, 
        multem::Input input, 
        py::iterator iterator) {

    // Call the simulate method
    return multem::simulate_slices(
        config, 
        input, 
        make_slice_iterator<Slice>(iterator), 
        make_slice_iterator<Slice>(iterator.sentinel()));
  }
  

  /**
   * Type cast a multem::Atom object to a tuple
   */
  template <> 
  class type_caster<multem::Atom> {
  public:
  
    PYBIND11_TYPE_CASTER(multem::Atom, _("multem::Atom"));

    bool load(handle src, bool convert) {
      if (py::isinstance<py::tuple>(src)) {
        py::tuple t = py::cast<py::tuple>(src);
        if (py::len(t) == 8) {
          value.element = py::cast<int>(t[0]);
          value.x = py::cast<double>(t[1]);
          value.y = py::cast<double>(t[2]);
          value.z = py::cast<double>(t[3]);
          value.sigma = py::cast<double>(t[4]);
          value.occupancy = py::cast<double>(t[5]);
          value.region = py::cast<int>(t[6]);
          value.charge = py::cast<int>(t[7]);
          return true;
        }
      }
      return false;
    }

    static handle cast(multem::Atom src, return_value_policy policy, handle parent) {
      return py::make_tuple(
        src.element, 
        src.x, 
        src.y, 
        src.z, 
        src.sigma, 
        src.occupancy, 
        src.region, 
        src.charge).release();
    }
  };

  
  /**
   * Type cast a multem::AmorphousLayer object to a tuple
   */
  template <> 
  class type_caster<multem::AmorphousLayer> {
  public:
  
    PYBIND11_TYPE_CASTER(multem::AmorphousLayer, _("multem::AmorphousLayer"));

    bool load(handle src, bool convert) {
      if (py::isinstance<py::tuple>(src)) {
        py::tuple t = py::cast<py::tuple>(src);
        if (py::len(t) == 3) {
          value.z_0 = py::cast<double>(t[0]);
          value.z_e = py::cast<double>(t[1]);
          value.dz = py::cast<double>(t[2]);
          return true;
        }
      }
      return false;
    }

    static handle cast(multem::AmorphousLayer src, return_value_policy policy, handle parent) {
      return py::make_tuple(
        src.z_0, 
        src.z_e, 
        src.dz).release();
    }
  };
 
  /**
   * Type cast a multem::STEMDetector::Angles object to a tuple
   */
  template <> 
  class type_caster<multem::STEMDetector::Angles> {
  public:
  
    PYBIND11_TYPE_CASTER(multem::STEMDetector::Angles, _("multem::STEMDetector::Angles"));

    bool load(handle src, bool convert) {
      if (py::isinstance<py::tuple>(src)) {
        py::tuple t = py::cast<py::tuple>(src);
        if (py::len(t) == 2) {
          value.inner_ang = py::cast<double>(t[0]);
          value.outer_ang = py::cast<double>(t[1]);
          return true;
        }
      }
      return false;
    }

    static handle cast(multem::STEMDetector::Angles src, return_value_policy policy, handle parent) {
      return py::make_tuple(
        src.inner_ang, 
        src.outer_ang).release();
    }
  };
  
  /**
   * Type cast a multem::STEMDetector::Radial object to a tuple
   */
  template <> 
  class type_caster<multem::STEMDetector::Radial> {
  public:
  
    PYBIND11_TYPE_CASTER(multem::STEMDetector::Radial, _("multem::STEMDetector::Radial"));

    bool load(handle src, bool convert) {
      if (py::isinstance<py::tuple>(src)) {
        py::tuple t = py::cast<py::tuple>(src);
        if (py::len(t) == 2) {
          value.x = py::cast<double>(t[0]);
          value.fx = py::cast<std::vector<double>>(t[1]);
          return true;
        }
      }
      return false;
    }

    static handle cast(multem::STEMDetector::Radial src, return_value_policy policy, handle parent) {
      return py::make_tuple(
        src.x, 
        src.fx).release();
    }
  };
  
  /**
   * Type cast a multem::STEMDetector::Matrix object to a tuple
   */
  template <> 
  class type_caster<multem::STEMDetector::Matrix> {
  public:
  
    PYBIND11_TYPE_CASTER(multem::STEMDetector::Matrix, _("multem::STEMDetector::Matrix"));

    bool load(handle src, bool convert) {
      if (py::isinstance<py::tuple>(src)) {
        py::tuple t = py::cast<py::tuple>(src);
        if (py::len(t) == 2) {
          value.R = py::cast<double>(t[0]);
          value.fR = py::cast<std::vector<double>>(t[1]);
          return true;
        }
      }
      return false;
    }

    static handle cast(multem::STEMDetector::Matrix src, return_value_policy policy, handle parent) {
      return py::make_tuple(
        src.R, 
        src.fR).release();
    }
  };
  

  /**
   * Type cast a Slice object to a tuple
   */
  template <> 
  class type_caster<Slice> {
  public:
  
    PYBIND11_TYPE_CASTER(Slice, _("Slice"));

    bool load(handle src, bool convert) {
      if (py::isinstance<py::tuple>(src)) {
        py::tuple t = py::cast<py::tuple>(src);
        if (py::len(t) == 3) {
          value.spec_z0 = py::cast<double>(t[0]);
          value.spec_lz = py::cast<double>(t[1]);
          value.spec_atoms = py::cast< std::vector<multem::Atom> >(t[2]);
          return true;
        }
      }
      return false;
    }

    static handle cast(Slice src, return_value_policy policy, handle parent) {
      return py::make_tuple(
        src.spec_z0, 
        src.spec_lz, 
        src.spec_atoms).release();
    }
  };


  /**
   * Get the py::buffer_info from a std::vector
   * @param self A std::vector
   * @returns A py::buffer_info object
   */
  template <typename T>
  py::buffer_info as_buffer_info(std::vector<T> &self) {
    return py::buffer_info(
      self.data(),
      sizeof(T),
      py::format_descriptor<T>::format(),
      1,
      { self.size() },
      { sizeof(T) });
  }
  
  /**
   * Get the py::buffer_info from a multem::Image
   * @param self A multem::Image
   * @returns A py::buffer_info object
   */
  template <typename T>
  py::buffer_info as_buffer_info(multem::Image<T> &self) {
    typedef typename multem::Image<T>::value_type value_type;
    return py::buffer_info(
        self.data.data(), 
        sizeof(value_type), 
        py::format_descriptor<value_type>::format(),
        2,
        { 
          self.shape[0], 
          self.shape[1] 
        },
        { 
          sizeof(value_type) * self.shape[1],
          sizeof(value_type) 
        });
  }


  /**
   * Get the py::array_t from a std::vector
   * @param self A std::vector
   * @returns A py::array_t object
   */
  template <typename T>
  py::array_t<T> as_array_t(std::vector<T> &self) {
    return py::array_t<T>(as_buffer_info(self));
  }

  /**
   * Get the py::array_t from a multem::Image
   * @param self A multem::Image
   * @returns A py::array_t object
   */
  template <typename T>
  py::array_t<T> as_array_t(multem::Image<T> &self) {
    return py::array_t<T>(as_buffer_info(self));
  }

  /**
   * Define helper functions for a type
   */
  template <typename T>
  struct Helpers {};

  /**
   * Define helper function for the multem::SystemConfiguration class
   */
  template <>
  struct Helpers <multem::SystemConfiguration> {
   
    /**
     * Get the state
     */
    static py::tuple getstate(const multem::SystemConfiguration &self) {
      return py::make_tuple(
        self.device,
        self.precision,
        self.cpu_ncores,
        self.cpu_nthread,
        self.gpu_device,
        self.gpu_nstream);
    }

    /**
     * Set the state
     */
    static multem::SystemConfiguration setstate(py::tuple obj) {
      multem::SystemConfiguration self;
      self.device = obj[0].cast<std::string>();
      self.precision = obj[1].cast<std::string>();
      self.cpu_ncores = obj[2].cast<std::size_t>();
      self.cpu_nthread = obj[3].cast<std::size_t>();
      self.gpu_device = obj[4].cast<std::size_t>();
      self.gpu_nstream = obj[5].cast<std::size_t>();
      return self;
    }
  };
  
  /**
   * Define helper function for the multem::STEMDetector class
   */
  template <>
  struct Helpers <multem::STEMDetector> {
   
    /**
     * Get the state
     */
    static py::tuple getstate(const multem::STEMDetector &self) {
      return py::make_tuple(
        self.type,
        self.cir,
        self.radial,
        self.matrix);
    }

    /**
     * Set the state
     */
    static multem::STEMDetector setstate(py::tuple obj) {
      multem::STEMDetector self;
      self.type = obj[0].cast<std::string>();
      self.cir = obj[1].cast< std::vector<multem::STEMDetector::Angles> >();
      self.radial = obj[2].cast< std::vector<multem::STEMDetector::Radial> >();
      self.matrix = obj[3].cast< std::vector<multem::STEMDetector::Matrix> >();
      return self;
    }
  };

  /**
   * Define helper functions for the multem::Input class
   */
  template <>
  struct Helpers <multem::Input> {
    
    /**
     * Get the state
     */
    static py::tuple getstate(const multem::Input &self) {
      return py::make_tuple(
        self.interaction_model, 
        self.potential_type,  
        self.operation_mode,
        self.memory_size,
        self.reverse_multislice,
        self.pn_model,
        self.pn_coh_contrib,
        self.pn_single_conf,
        self.pn_nconf,
        self.pn_dim,
        self.pn_seed,
        py::list(py::cast(self.spec_atoms)),
        self.spec_dz,
        self.spec_lx,
        self.spec_ly,
        self.spec_lz,
        self.spec_cryst_na,
        self.spec_cryst_nb,
        self.spec_cryst_nc,
        self.spec_cryst_a,
        self.spec_cryst_b,
        self.spec_cryst_c,
        self.spec_cryst_x0,
        self.spec_cryst_y0,
        self.spec_amorp,
        self.spec_rot_theta,
        self.spec_rot_u0,
        self.spec_rot_center_type,
        self.spec_rot_center_p,
        self.thick_type,
        self.thick,
        self.potential_slicing,
        self.nx,
        self.ny,
        self.bwl,
        self.simulation_type,
        self.iw_type,
        self.iw_psi,
        self.iw_x,
        self.iw_y,
        self.E_0,
        self.theta,
        self.phi,
        self.illumination_model,
        self.temporal_spatial_incoh,
        self.cond_lens_m,
        self.cond_lens_c_10,
        self.cond_lens_c_12,
        self.cond_lens_phi_12,
        self.cond_lens_c_21,
        self.cond_lens_phi_21,
        self.cond_lens_c_23,
        self.cond_lens_phi_23,
        self.cond_lens_c_30,
        self.cond_lens_c_32,
        self.cond_lens_phi_32,
        self.cond_lens_c_34,
        self.cond_lens_phi_34,
        self.cond_lens_c_41,
        self.cond_lens_phi_41,
        self.cond_lens_c_43,
        self.cond_lens_phi_43,
        self.cond_lens_c_45,
        self.cond_lens_phi_45,
        self.cond_lens_c_50,
        self.cond_lens_c_52,
        self.cond_lens_phi_52,
        self.cond_lens_c_54,
        self.cond_lens_phi_54,
        self.cond_lens_c_56,
        self.cond_lens_phi_56,
        self.cond_lens_inner_aper_ang,
        self.cond_lens_outer_aper_ang,
        self.cond_lens_si_sigma,
        self.cond_lens_si_a,
        self.cond_lens_si_beta,
        self.cond_lens_si_rad_npts,
        self.cond_lens_si_azm_npts,
        self.cond_lens_ti_sigma,
        self.cond_lens_ti_a,
        self.cond_lens_ti_beta,
        self.cond_lens_ti_npts,
        self.cond_lens_zero_defocus_type,
        self.cond_lens_zero_defocus_plane,
        self.obj_lens_m,
        self.obj_lens_c_10,
        self.obj_lens_c_12,
        self.obj_lens_phi_12,
        self.obj_lens_c_21,
        self.obj_lens_phi_21,
        self.obj_lens_c_23,
        self.obj_lens_phi_23,
        self.obj_lens_c_30,
        self.obj_lens_c_32,
        self.obj_lens_phi_32,
        self.obj_lens_c_34,
        self.obj_lens_phi_34,
        self.obj_lens_c_41,
        self.obj_lens_phi_41,
        self.obj_lens_c_43,
        self.obj_lens_phi_43,
        self.obj_lens_c_45,
        self.obj_lens_phi_45,
        self.obj_lens_c_50,
        self.obj_lens_c_52,
        self.obj_lens_phi_52,
        self.obj_lens_c_54,
        self.obj_lens_phi_54,
        self.obj_lens_c_56,
        self.obj_lens_phi_56,
        self.obj_lens_inner_aper_ang,
        self.obj_lens_outer_aper_ang,
        self.obj_lens_ti_sigma,
        self.obj_lens_ti_npts,
        self.obj_lens_zero_defocus_type,
        self.obj_lens_zero_defocus_plane,
        self.phase_shift,
        self.detector,
        self.scanning_type,
        self.scanning_square_pxs,
        self.scanning_periodic,
        self.scanning_ns,
        self.scanning_x0,
        self.scanning_y0,
        self.scanning_xe,
        self.scanning_ye,
        self.ped_nrot,
        self.ped_theta,
        self.hci_nrot,
        self.hci_theta,
        self.eels_Z,
        self.eels_E_loss,
        self.eels_collection_angle,
        self.eels_m_selection,
        self.eels_channelling_type,
        self.eftem_Z,
        self.eftem_E_loss,
        self.eftem_collection_angle,
        self.eftem_m_selection,
        self.eftem_channelling_type,
        self.output_area_ix_0,
        self.output_area_iy_0,
        self.output_area_ix_e,
        self.output_area_iy_e);
    }

    /**
     * Set the state
     */
    static multem::Input setstate(py::tuple obj) {
      multem::Input self;
      self.interaction_model = obj[0].cast< std::string >(); 
      self.potential_type = obj[1].cast< std::string >();  
      self.operation_mode = obj[2].cast< std::string >();
      self.memory_size = obj[3].cast< std::size_t >();
      self.reverse_multislice = obj[4].cast< bool >();
      self.pn_model = obj[5].cast< std::string >();
      self.pn_coh_contrib = obj[6].cast< bool >();
      self.pn_single_conf = obj[7].cast< bool >();
      self.pn_nconf = obj[8].cast< int >();
      self.pn_dim = obj[9].cast< int >();
      self.pn_seed = obj[10].cast< int >();
      py::list spec_atoms = obj[11].cast< py::list >();
      self.spec_atoms.reserve(spec_atoms.size());
      for (auto x : spec_atoms) {
        self.spec_atoms.push_back(x.cast<multem::Atom>());
      }
      self.spec_dz = obj[12].cast< double >();
      self.spec_lx = obj[13].cast< double >();
      self.spec_ly = obj[14].cast< double >();
      self.spec_lz = obj[15].cast< double >();
      self.spec_cryst_na = obj[16].cast< int >();
      self.spec_cryst_nb = obj[17].cast< int >();
      self.spec_cryst_nc = obj[18].cast< int >();
      self.spec_cryst_a = obj[19].cast< double >();
      self.spec_cryst_b = obj[20].cast< double >();
      self.spec_cryst_c = obj[21].cast< double >();
      self.spec_cryst_x0 = obj[22].cast< double >();
      self.spec_cryst_y0 = obj[23].cast< double >();
      self.spec_amorp = obj[24].cast< std::vector<multem::AmorphousLayer> >();
      self.spec_rot_theta = obj[25].cast< double >();
      self.spec_rot_u0 = obj[26].cast< std::array<double, 3> >();
      self.spec_rot_center_type = obj[27].cast< std::string >();
      self.spec_rot_center_p = obj[28].cast< std::array<double, 3> >();
      self.thick_type = obj[29].cast< std::string >();
      self.thick = obj[30].cast< std::vector<double> >();
      self.potential_slicing = obj[31].cast< std::string >();
      self.nx = obj[32].cast< int >();
      self.ny = obj[33].cast< int >();
      self.bwl = obj[34].cast< bool >();
      self.simulation_type = obj[35].cast< std::string >();
      self.iw_type = obj[36].cast< std::string >();
      self.iw_psi = obj[37].cast< std::vector< std::complex<double> > >();
      self.iw_x = obj[38].cast< std::vector<double> >();
      self.iw_y = obj[39].cast< std::vector<double> >();
      self.E_0 = obj[40].cast< double >();
      self.theta = obj[41].cast< double >();
      self.phi = obj[42].cast< double >();
      self.illumination_model = obj[43].cast< std::string >();
      self.temporal_spatial_incoh = obj[44].cast< std::string >();
      self.cond_lens_m = obj[45].cast< int >();
      self.cond_lens_c_10 = obj[46].cast< double >();
      self.cond_lens_c_12 = obj[47].cast< double >();
      self.cond_lens_phi_12 = obj[48].cast< double >();
      self.cond_lens_c_21 = obj[49].cast< double >();
      self.cond_lens_phi_21 = obj[50].cast< double >();
      self.cond_lens_c_23 = obj[51].cast< double >();
      self.cond_lens_phi_23 = obj[52].cast< double >();
      self.cond_lens_c_30 = obj[53].cast< double >();
      self.cond_lens_c_32 = obj[54].cast< double >();
      self.cond_lens_phi_32 = obj[55].cast< double >();
      self.cond_lens_c_34 = obj[56].cast< double >();
      self.cond_lens_phi_34 = obj[57].cast< double >();
      self.cond_lens_c_41 = obj[58].cast< double >();
      self.cond_lens_phi_41 = obj[59].cast< double >();
      self.cond_lens_c_43 = obj[60].cast< double >();
      self.cond_lens_phi_43 = obj[61].cast< double >();
      self.cond_lens_c_45 = obj[62].cast< double >();
      self.cond_lens_phi_45 = obj[63].cast< double >();
      self.cond_lens_c_50 = obj[64].cast< double >();
      self.cond_lens_c_52 = obj[65].cast< double >();
      self.cond_lens_phi_52 = obj[66].cast< double >();
      self.cond_lens_c_54 = obj[67].cast< double >();
      self.cond_lens_phi_54 = obj[68].cast< double >();
      self.cond_lens_c_56 = obj[69].cast< double >();
      self.cond_lens_phi_56 = obj[70].cast< double >();
      self.cond_lens_inner_aper_ang = obj[71].cast< double >();
      self.cond_lens_outer_aper_ang = obj[72].cast< double >();
      self.cond_lens_si_sigma = obj[73].cast< double >();
      self.cond_lens_si_a = obj[74].cast<double>();
      self.cond_lens_si_beta = obj[75].cast<double>();
      self.cond_lens_si_rad_npts = obj[76].cast<int>();
      self.cond_lens_si_azm_npts = obj[77].cast<int>();
      self.cond_lens_ti_sigma = obj[78].cast< double >();
      self.cond_lens_ti_a = obj[79].cast<double>();
      self.cond_lens_ti_beta = obj[80].cast<double>();
      self.cond_lens_ti_npts = obj[81].cast< int >();
      self.cond_lens_zero_defocus_type = obj[82].cast< std::string >();
      self.cond_lens_zero_defocus_plane = obj[83].cast< double >();
      self.obj_lens_m = obj[84].cast< int >();
      self.obj_lens_c_10 = obj[85].cast< double >();
      self.obj_lens_c_12 = obj[86].cast< double >();
      self.obj_lens_phi_12 = obj[87].cast< double >();
      self.obj_lens_c_21 = obj[88].cast< double >();
      self.obj_lens_phi_21 = obj[89].cast< double >();
      self.obj_lens_c_23 = obj[90].cast< double >();
      self.obj_lens_phi_23 = obj[91].cast< double >();
      self.obj_lens_c_30 = obj[92].cast< double >();
      self.obj_lens_c_32 = obj[93].cast< double >();
      self.obj_lens_phi_32 = obj[94].cast< double >();
      self.obj_lens_c_34 = obj[95].cast< double >();
      self.obj_lens_phi_34 = obj[96].cast< double >();
      self.obj_lens_c_41 = obj[97].cast< double >();
      self.obj_lens_phi_41 = obj[98].cast< double >();
      self.obj_lens_c_43 = obj[99].cast< double >();
      self.obj_lens_phi_43 = obj[100].cast< double >();
      self.obj_lens_c_45 = obj[101].cast< double >();
      self.obj_lens_phi_45 = obj[102].cast< double >();
      self.obj_lens_c_50 = obj[103].cast< double >();
      self.obj_lens_c_52 = obj[104].cast< double >();
      self.obj_lens_phi_52 = obj[105].cast< double >();
      self.obj_lens_c_54 = obj[106].cast< double >();
      self.obj_lens_phi_54 = obj[107].cast< double >();
      self.obj_lens_c_56 = obj[108].cast< double >();
      self.obj_lens_phi_56 = obj[109].cast< double >();
      self.obj_lens_inner_aper_ang = obj[110].cast< double >();
      self.obj_lens_outer_aper_ang = obj[111].cast< double >();
      self.obj_lens_ti_sigma = obj[112].cast< double >();
      self.obj_lens_ti_npts = obj[113].cast< int >();
      self.obj_lens_zero_defocus_type = obj[114].cast< std::string >();
      self.obj_lens_zero_defocus_plane = obj[115].cast< double >();
      self.phase_shift = obj[116].cast<double>();
      self.detector = obj[117].cast< multem::STEMDetector >();
      self.scanning_type = obj[118].cast< std::string >();
      self.scanning_square_pxs = obj[119].cast< bool >();
      self.scanning_periodic = obj[120].cast< bool >();
      self.scanning_ns = obj[121].cast< int >();
      self.scanning_x0 = obj[122].cast< double >();
      self.scanning_y0 = obj[123].cast< double >();
      self.scanning_xe = obj[124].cast< double >();
      self.scanning_ye = obj[125].cast< double >();
      self.ped_nrot = obj[126].cast< double >();
      self.ped_theta = obj[127].cast< double >();
      self.hci_nrot = obj[128].cast< double >();
      self.hci_theta = obj[129].cast< double >();
      self.eels_Z = obj[130].cast< int >();
      self.eels_E_loss = obj[131].cast< double >();
      self.eels_collection_angle = obj[132].cast< double >();
      self.eels_m_selection = obj[133].cast< int >();
      self.eels_channelling_type = obj[134].cast< std::string >();
      self.eftem_Z = obj[135].cast< int >();
      self.eftem_E_loss = obj[136].cast< double >();
      self.eftem_collection_angle = obj[137].cast< double >();
      self.eftem_m_selection = obj[138].cast< int >();
      self.eftem_channelling_type = obj[139].cast< std::string >();
      self.output_area_ix_0 = obj[140].cast< int >();
      self.output_area_iy_0 = obj[141].cast< int >();
      self.output_area_ix_e = obj[142].cast< int >();
      self.output_area_iy_e = obj[143].cast< int >();
      return self;
    }

    /**
     * Convert the object to a python dictionary
     * @param self The object
     * @returns A python dictionary
     */
    static py::dict asdict(multem::Input &self) {
      py::dict result;
      result["interaction_model"] = self.interaction_model; 
      result["potential_type"] = self.potential_type;  
      result["operation_mode"] = self.operation_mode;
      result["memory_size"] = self.memory_size;
      result["reverse_multislice"] = self.reverse_multislice;
      result["pn_model"] = self.pn_model;
      result["pn_coh_contrib"] = self.pn_coh_contrib;
      result["pn_single_conf"] = self.pn_single_conf;
      result["pn_nconf"] = self.pn_nconf;
      result["pn_dim"] = self.pn_dim;
      result["pn_seed"] = self.pn_seed;
      
      py::list spec_atoms;
      for (auto item : self.spec_atoms) {
        spec_atoms.append(item);
      }
      result["spec_atoms"] = spec_atoms;
      
      result["spec_dz"] = self.spec_dz;
      result["spec_lx"] = self.spec_lx;
      result["spec_ly"] = self.spec_ly;
      result["spec_lz"] = self.spec_lz;
      result["spec_cryst_na"] = self.spec_cryst_na;
      result["spec_cryst_nb"] = self.spec_cryst_nb;
      result["spec_cryst_nc"] = self.spec_cryst_nc;
      result["spec_cryst_a"] = self.spec_cryst_a;
      result["spec_cryst_b"] = self.spec_cryst_b;
      result["spec_cryst_c"] = self.spec_cryst_c;
      result["spec_cryst_x0"] = self.spec_cryst_x0;
      result["spec_cryst_y0"] = self.spec_cryst_y0;

      py::list spec_amorp;
      for (auto item : self.spec_amorp) {
        spec_amorp.append(item);
      }
      result["spec_amorp"] = spec_amorp;
      
      result["spec_rot_theta"] = self.spec_rot_theta;
      result["spec_rot_u0"] = self.spec_rot_u0;
      result["spec_rot_center_type"] = self.spec_rot_center_type;
      result["spec_rot_center_p"] = self.spec_rot_center_p;
      result["thick_type"] = self.thick_type;
      result["thick"] = py::detail::as_array_t(self.thick);
      result["potential_slicing"] = self.potential_slicing;
      result["nx"] = self.nx;
      result["ny"] = self.ny;
      result["bwl"] = self.bwl;
      result["simulation_type"] = self.simulation_type;
      result["iw_type"] = self.iw_type;
      result["iw_psi"] = py::detail::as_array_t(self.iw_psi);
      result["iw_x"] = py::detail::as_array_t(self.iw_x);
      result["iw_y"] = py::detail::as_array_t(self.iw_y);
      result["E_0"] = self.E_0;
      result["theta"] = self.theta;
      result["phi"] = self.phi;
      result["illumination_model"] = self.illumination_model;
      result["temporal_spatial_incoh"] = self.temporal_spatial_incoh;
      result["cond_lens_m"] = self.cond_lens_m;
      result["cond_lens_c_10"] = self.cond_lens_c_10;
      result["cond_lens_c_12"] = self.cond_lens_c_12;
      result["cond_lens_phi_12"] = self.cond_lens_phi_12;
      result["cond_lens_c_21"] = self.cond_lens_c_21;
      result["cond_lens_phi_21"] = self.cond_lens_phi_21;
      result["cond_lens_c_23"] = self.cond_lens_c_23;
      result["cond_lens_phi_23"] = self.cond_lens_phi_23;
      result["cond_lens_c_30"] = self.cond_lens_c_30;
      result["cond_lens_c_32"] = self.cond_lens_c_32;
      result["cond_lens_phi_32"] = self.cond_lens_phi_32;
      result["cond_lens_c_34"] = self.cond_lens_c_34;
      result["cond_lens_phi_34"] = self.cond_lens_phi_34;
      result["cond_lens_c_41"] = self.cond_lens_c_41;
      result["cond_lens_phi_41"] = self.cond_lens_phi_41;
      result["cond_lens_c_43"] = self.cond_lens_c_43;
      result["cond_lens_phi_43"] = self.cond_lens_phi_43;
      result["cond_lens_c_45"] = self.cond_lens_c_45;
      result["cond_lens_phi_45"] = self.cond_lens_phi_45;
      result["cond_lens_c_50"] = self.cond_lens_c_50;
      result["cond_lens_c_52"] = self.cond_lens_c_52;
      result["cond_lens_phi_52"] = self.cond_lens_phi_52;
      result["cond_lens_c_54"] = self.cond_lens_c_54;
      result["cond_lens_phi_54"] = self.cond_lens_phi_54;
      result["cond_lens_c_56"] = self.cond_lens_c_56;
      result["cond_lens_phi_56"] = self.cond_lens_phi_56;
      result["cond_lens_inner_aper_ang"] = self.cond_lens_inner_aper_ang;
      result["cond_lens_outer_aper_ang"] = self.cond_lens_outer_aper_ang;
      result["cond_lens_si_sigma"] = self.cond_lens_si_sigma;
      result["cond_lens_si_a"] = self.cond_lens_si_a;
      result["cond_lens_si_beta"] = self.cond_lens_si_beta;
      result["cond_lens_si_rad_npts"] = self.cond_lens_si_rad_npts;
      result["cond_lens_si_azm_npts"] = self.cond_lens_si_azm_npts;
      result["cond_lens_ti_sigma"] = self.cond_lens_ti_sigma;
      result["cond_lens_ti_a"] = self.cond_lens_ti_a;
      result["cond_lens_ti_beta"] = self.cond_lens_ti_beta;
      result["cond_lens_ti_npts"] = self.cond_lens_ti_npts;
      result["cond_lens_zero_defocus_type"] = self.cond_lens_zero_defocus_type;
      result["cond_lens_zero_defocus_plane"] = self.cond_lens_zero_defocus_plane;
      result["obj_lens_m"] = self.obj_lens_m;
      result["obj_lens_c_10"] = self.obj_lens_c_10;
      result["obj_lens_c_12"] = self.obj_lens_c_12;
      result["obj_lens_phi_12"] = self.obj_lens_phi_12;
      result["obj_lens_c_21"] = self.obj_lens_c_21;
      result["obj_lens_phi_21"] = self.obj_lens_phi_21;
      result["obj_lens_c_23"] = self.obj_lens_c_23;
      result["obj_lens_phi_23"] = self.obj_lens_phi_23;
      result["obj_lens_c_30"] = self.obj_lens_c_30;
      result["obj_lens_c_32"] = self.obj_lens_c_32;
      result["obj_lens_phi_32"] = self.obj_lens_phi_32;
      result["obj_lens_c_34"] = self.obj_lens_c_34;
      result["obj_lens_phi_34"] = self.obj_lens_phi_34;
      result["obj_lens_c_41"] = self.obj_lens_c_41;
      result["obj_lens_phi_41"] = self.obj_lens_phi_41;
      result["obj_lens_c_43"] = self.obj_lens_c_43;
      result["obj_lens_phi_43"] = self.obj_lens_phi_43;
      result["obj_lens_c_45"] = self.obj_lens_c_45;
      result["obj_lens_phi_45"] = self.obj_lens_phi_45;
      result["obj_lens_c_50"] = self.obj_lens_c_50;
      result["obj_lens_c_52"] = self.obj_lens_c_52;
      result["obj_lens_phi_52"] = self.obj_lens_phi_52;
      result["obj_lens_c_54"] = self.obj_lens_c_54;
      result["obj_lens_phi_54"] = self.obj_lens_phi_54;
      result["obj_lens_c_56"] = self.obj_lens_c_56;
      result["obj_lens_phi_56"] = self.obj_lens_phi_56;
      result["obj_lens_inner_aper_ang"] = self.obj_lens_inner_aper_ang;
      result["obj_lens_outer_aper_ang"] = self.obj_lens_outer_aper_ang;
      result["obj_lens_ti_sigma"] = self.obj_lens_ti_sigma;
      result["obj_lens_ti_npts"] = self.obj_lens_ti_npts;
      result["obj_lens_zero_defocus_type"] = self.obj_lens_zero_defocus_type;
      result["obj_lens_zero_defocus_plane"] = self.obj_lens_zero_defocus_plane;
      result["phase_shift"] = self.phase_shift;
      //STEMDetector detector;
      result["scanning_type"] = self.scanning_type;
      result["scanning_square_pxs"] = self.scanning_square_pxs;
      result["scanning_periodic"] = self.scanning_periodic;
      result["scanning_ns"] = self.scanning_ns;
      result["scanning_x0"] = self.scanning_x0;
      result["scanning_y0"] = self.scanning_y0;
      result["scanning_xe"] = self.scanning_xe;
      result["scanning_ye"] = self.scanning_ye;
      result["ped_nrot"] = self.ped_nrot;
      result["ped_theta"] = self.ped_theta;
      result["hci_nrot"] = self.hci_nrot;
      result["hci_theta"] = self.hci_theta;
      result["eels_Z"] = self.eels_Z;
      result["eels_E_loss"] = self.eels_E_loss;
      result["eels_collection_angle"] = self.eels_collection_angle;
      result["eels_m_selection"] = self.eels_m_selection;
      result["eels_channelling_type"] = self.eels_channelling_type;
      result["eftem_Z"] = self.eftem_Z;
      result["eftem_E_loss"] = self.eftem_E_loss;
      result["eftem_collection_angle"] = self.eftem_collection_angle;
      result["eftem_m_selection"] = self.eftem_m_selection;
      result["eftem_channelling_type"] = self.eftem_channelling_type;
      result["output_area_ix_0"] = self.output_area_ix_0;
      result["output_area_iy_0"] = self.output_area_iy_0;
      result["output_area_ix_e"] = self.output_area_ix_e;
      result["output_area_iy_e"] = self.output_area_iy_e;
      return result;
    }
  };

  /**
   * Define helper functions for the multem::Image class
   */
  template<typename T>
  struct Helpers < multem::Image<T> > {

    /**
     * Create a multem::Image from a py::array_t
     * @param array The py::array_t object
     * @returns The multem::Image object
     */
    static multem::Image<T> init_from_array_t(py::array_t<T> array) {
      py::buffer_info buffer = array.request();
      MULTEM_ASSERT(buffer.ndim == 2);
      MULTEM_ASSERT(buffer.shape[0] >= 0);
      MULTEM_ASSERT(buffer.shape[1] >= 0);
      return multem::Image<T>(
        (T *) buffer.ptr, 
        typename multem::Image<T>::shape_type({
          (std::size_t) buffer.shape[0], 
          (std::size_t) buffer.shape[1]}));
    }

    /**
     * Create a py::buffer_info object from a multem::Image object
     * @param self The multem::Image object
     * @returns The py::buffer_info object
     */
    static py::buffer_info as_buffer_info(multem::Image<T> &self) {
      return py::detail::as_buffer_info(self);
    }
  };

  /**
   * Define helper functions for the multem::Data class
   */
  template <>
  struct Helpers <multem::Data> {
    
    /**
     * Convert the object to a python dictionary
     * @param self The object
     * @returns A python dictionary
     */
    static py::dict asdict(multem::Data &self) {
      py::dict result;
      result["m2psi_tot"] = py::detail::as_array_t<double>(self.m2psi_tot);
      result["m2psi_coh"] = py::detail::as_array_t<double>(self.m2psi_coh);
      result["psi_coh"] = py::detail::as_array_t< std::complex<double> >(self.psi_coh);
      return result;
    }
  };
  
  /**
   * Define helper functions for the multem::Output class
   */
  template <>
  struct Helpers <multem::Output> {
    
    /**
     * Convert the object to a python dictionary
     * @param self The object
     * @returns A python dictionary
     */
    static py::dict asdict(multem::Output &self) {
      py::dict result;
      result["dx"] = self.dx;
      result["dy"] = self.dy;
      result["x"] = py::detail::as_array_t<double>(self.x);
      result["y"] = py::detail::as_array_t<double>(self.y);
      result["thick"] = py::detail::as_array_t<double>(self.thick);
      py::list data;
      for (auto item : self.data) {
        data.append(Helpers<multem::Data>::asdict(item));
      }
      result["data"] = data;
      return result;
    }
  };

  /**
   * Wrap a multem::Image<T> class as a buffer object
   */
  template <typename T>
  py::class_< multem::Image<T> > image_class(py::module &m, const char *name) {
    return py::class_< multem::Image<T> >(m, name, py::buffer_protocol())
      .def(py::init<>())
      .def(py::init(&py::detail::Helpers<multem::Image<T>>::init_from_array_t))
      .def_buffer(&py::detail::Helpers<multem::Image<T>>::as_buffer_info)
      ;
  }

}}

PYBIND11_MODULE(multem_ext, m)
{

  // Wrap the vector of atoms
  py::bind_vector<std::vector<multem::Atom>>(m, "AtomList");

  // Wrap the multem::CrystalParameters class
  py::class_<multem::CrystalParameters>(m, "CrystalParameters")
    .def(py::init<>())
    .def_readwrite("na", &multem::CrystalParameters::na)
    .def_readwrite("nb", &multem::CrystalParameters::nb)
    .def_readwrite("nc", &multem::CrystalParameters::nc)
    .def_readwrite("a", &multem::CrystalParameters::a)
    .def_readwrite("b", &multem::CrystalParameters::b)
    .def_readwrite("c", &multem::CrystalParameters::c)
    .def_readwrite("layers", &multem::CrystalParameters::layers)
    ;
 
  // Wrap the multem::STEMDetector class
  py::class_<multem::STEMDetector>(m, "STEMDetector")
    .def(py::init<>())
    .def_readwrite("type", &multem::STEMDetector::type)
    .def_readwrite("cir", &multem::STEMDetector::cir)
    .def_readwrite("radial", &multem::STEMDetector::radial)
    .def_readwrite("matrix", &multem::STEMDetector::matrix)
    .def(py::pickle(
        &py::detail::Helpers<multem::STEMDetector>::getstate,
        &py::detail::Helpers<multem::STEMDetector>::setstate))
    ;

  // Wrap the multem::Input class
  py::class_<multem::Input>(m, "Input")
    .def(py::init<>())
    .def_readwrite("interaction_model", &multem::Input::interaction_model) 
    .def_readwrite("potential_type", &multem::Input::potential_type)  
    .def_readwrite("operation_mode", &multem::Input::operation_mode)
    .def_readwrite("memory_size", &multem::Input::memory_size)
    .def_readwrite("reverse_multislice", &multem::Input::reverse_multislice)
    .def_readwrite("pn_model", &multem::Input::pn_model)
    .def_readwrite("pn_coh_contrib", &multem::Input::pn_coh_contrib)
    .def_readwrite("pn_single_conf", &multem::Input::pn_single_conf)
    .def_readwrite("pn_nconf", &multem::Input::pn_nconf)
    .def_readwrite("pn_dim", &multem::Input::pn_dim)
    .def_readwrite("pn_seed", &multem::Input::pn_seed)
    .def_readwrite("spec_atoms", &multem::Input::spec_atoms)
    .def_readwrite("spec_dz", &multem::Input::spec_dz)
    .def_readwrite("spec_lx", &multem::Input::spec_lx)
    .def_readwrite("spec_ly", &multem::Input::spec_ly)
    .def_readwrite("spec_lz", &multem::Input::spec_lz)
    .def_readwrite("spec_cryst_na", &multem::Input::spec_cryst_na)
    .def_readwrite("spec_cryst_nb", &multem::Input::spec_cryst_nb)
    .def_readwrite("spec_cryst_nc", &multem::Input::spec_cryst_nc)
    .def_readwrite("spec_cryst_a", &multem::Input::spec_cryst_a)
    .def_readwrite("spec_cryst_b", &multem::Input::spec_cryst_b)
    .def_readwrite("spec_cryst_c", &multem::Input::spec_cryst_c)
    .def_readwrite("spec_cryst_x0", &multem::Input::spec_cryst_x0)
    .def_readwrite("spec_cryst_y0", &multem::Input::spec_cryst_y0)
    .def_readwrite("spec_amorp", &multem::Input::spec_amorp)
    .def_readwrite("spec_rot_theta", &multem::Input::spec_rot_theta)
    .def_readwrite("spec_rot_u0", &multem::Input::spec_rot_u0)
    .def_readwrite("spec_rot_center_type", &multem::Input::spec_rot_center_type)
    .def_readwrite("spec_rot_center_p", &multem::Input::spec_rot_center_p)
    .def_readwrite("thick_type", &multem::Input::thick_type)
    .def_readwrite("thick", &multem::Input::thick)
    .def_readwrite("potential_slicing", &multem::Input::potential_slicing)
    .def_readwrite("static_B_factor", &multem::Input::static_B_factor)
    .def_readwrite("nx", &multem::Input::nx)
    .def_readwrite("ny", &multem::Input::ny)
    .def_readwrite("bwl", &multem::Input::bwl)
    .def_readwrite("simulation_type", &multem::Input::simulation_type)
    .def_readwrite("iw_type", &multem::Input::iw_type)
    .def_readwrite("iw_psi", &multem::Input::iw_psi)
    .def_readwrite("iw_x", &multem::Input::iw_x)
    .def_readwrite("iw_y", &multem::Input::iw_y)
    .def_readwrite("E_0", &multem::Input::E_0)
    .def_readwrite("theta", &multem::Input::theta)
    .def_readwrite("phi", &multem::Input::phi)
    .def_readwrite("illumination_model", &multem::Input::illumination_model)
    .def_readwrite("temporal_spatial_incoh", &multem::Input::temporal_spatial_incoh)
    .def_readwrite("cond_lens_m", &multem::Input::cond_lens_m)
    .def_readwrite("cond_lens_c_10", &multem::Input::cond_lens_c_10)
    .def_readwrite("cond_lens_c_12", &multem::Input::cond_lens_c_12)
    .def_readwrite("cond_lens_phi_12", &multem::Input::cond_lens_phi_12)
    .def_readwrite("cond_lens_c_21", &multem::Input::cond_lens_c_21)
    .def_readwrite("cond_lens_phi_21", &multem::Input::cond_lens_phi_21)
    .def_readwrite("cond_lens_c_23", &multem::Input::cond_lens_c_23)
    .def_readwrite("cond_lens_phi_23", &multem::Input::cond_lens_phi_23)
    .def_readwrite("cond_lens_c_30", &multem::Input::cond_lens_c_30)
    .def_readwrite("cond_lens_c_32", &multem::Input::cond_lens_c_32)
    .def_readwrite("cond_lens_phi_32", &multem::Input::cond_lens_phi_32)
    .def_readwrite("cond_lens_c_34", &multem::Input::cond_lens_c_34)
    .def_readwrite("cond_lens_phi_34", &multem::Input::cond_lens_phi_34)
    .def_readwrite("cond_lens_c_41", &multem::Input::cond_lens_c_41)
    .def_readwrite("cond_lens_phi_41", &multem::Input::cond_lens_phi_41)
    .def_readwrite("cond_lens_c_43", &multem::Input::cond_lens_c_43)
    .def_readwrite("cond_lens_phi_43", &multem::Input::cond_lens_phi_43)
    .def_readwrite("cond_lens_c_45", &multem::Input::cond_lens_c_45)
    .def_readwrite("cond_lens_phi_45", &multem::Input::cond_lens_phi_45)
    .def_readwrite("cond_lens_c_50", &multem::Input::cond_lens_c_50)
    .def_readwrite("cond_lens_c_52", &multem::Input::cond_lens_c_52)
    .def_readwrite("cond_lens_phi_52", &multem::Input::cond_lens_phi_52)
    .def_readwrite("cond_lens_c_54", &multem::Input::cond_lens_c_54)
    .def_readwrite("cond_lens_phi_54", &multem::Input::cond_lens_phi_54)
    .def_readwrite("cond_lens_c_56", &multem::Input::cond_lens_c_56)
    .def_readwrite("cond_lens_phi_56", &multem::Input::cond_lens_phi_56)
    .def_readwrite("cond_lens_inner_aper_ang", &multem::Input::cond_lens_inner_aper_ang)
    .def_readwrite("cond_lens_outer_aper_ang", &multem::Input::cond_lens_outer_aper_ang)
    .def_readwrite("cond_lens_si_sigma", &multem::Input::cond_lens_si_sigma)
    .def_readwrite("cond_lens_si_a", &multem::Input::cond_lens_si_a)
    .def_readwrite("cond_lens_si_beta", &multem::Input::cond_lens_si_beta)
    .def_readwrite("cond_lens_si_rad_npts", &multem::Input::cond_lens_si_rad_npts)
    .def_readwrite("cond_lens_si_azm_npts", &multem::Input::cond_lens_si_azm_npts)
    .def_readwrite("cond_lens_ti_sigma", &multem::Input::cond_lens_ti_sigma)
    .def_readwrite("cond_lens_ti_a", &multem::Input::cond_lens_ti_a)
    .def_readwrite("cond_lens_ti_beta", &multem::Input::cond_lens_ti_beta)
    .def_readwrite("cond_lens_ti_npts", &multem::Input::cond_lens_ti_npts)
    .def_readwrite("cond_lens_zero_defocus_type", &multem::Input::cond_lens_zero_defocus_type)
    .def_readwrite("cond_lens_zero_defocus_plane", &multem::Input::cond_lens_zero_defocus_plane)
    .def_readwrite("obj_lens_m", &multem::Input::obj_lens_m)
    .def_readwrite("obj_lens_c_10", &multem::Input::obj_lens_c_10)
    .def_readwrite("obj_lens_c_12", &multem::Input::obj_lens_c_12)
    .def_readwrite("obj_lens_phi_12", &multem::Input::obj_lens_phi_12)
    .def_readwrite("obj_lens_c_21", &multem::Input::obj_lens_c_21)
    .def_readwrite("obj_lens_phi_21", &multem::Input::obj_lens_phi_21)
    .def_readwrite("obj_lens_c_23", &multem::Input::obj_lens_c_23)
    .def_readwrite("obj_lens_phi_23", &multem::Input::obj_lens_phi_23)
    .def_readwrite("obj_lens_c_30", &multem::Input::obj_lens_c_30)
    .def_readwrite("obj_lens_c_32", &multem::Input::obj_lens_c_32)
    .def_readwrite("obj_lens_phi_32", &multem::Input::obj_lens_phi_32)
    .def_readwrite("obj_lens_c_34", &multem::Input::obj_lens_c_34)
    .def_readwrite("obj_lens_phi_34", &multem::Input::obj_lens_phi_34)
    .def_readwrite("obj_lens_c_41", &multem::Input::obj_lens_c_41)
    .def_readwrite("obj_lens_phi_41", &multem::Input::obj_lens_phi_41)
    .def_readwrite("obj_lens_c_43", &multem::Input::obj_lens_c_43)
    .def_readwrite("obj_lens_phi_43", &multem::Input::obj_lens_phi_43)
    .def_readwrite("obj_lens_c_45", &multem::Input::obj_lens_c_45)
    .def_readwrite("obj_lens_phi_45", &multem::Input::obj_lens_phi_45)
    .def_readwrite("obj_lens_c_50", &multem::Input::obj_lens_c_50)
    .def_readwrite("obj_lens_c_52", &multem::Input::obj_lens_c_52)
    .def_readwrite("obj_lens_phi_52", &multem::Input::obj_lens_phi_52)
    .def_readwrite("obj_lens_c_54", &multem::Input::obj_lens_c_54)
    .def_readwrite("obj_lens_phi_54", &multem::Input::obj_lens_phi_54)
    .def_readwrite("obj_lens_c_56", &multem::Input::obj_lens_c_56)
    .def_readwrite("obj_lens_phi_56", &multem::Input::obj_lens_phi_56)
    .def_readwrite("obj_lens_inner_aper_ang", &multem::Input::obj_lens_inner_aper_ang)
    .def_readwrite("obj_lens_outer_aper_ang", &multem::Input::obj_lens_outer_aper_ang)
    .def_readwrite("obj_lens_ti_a", &multem::Input::obj_lens_ti_a)
    .def_readwrite("obj_lens_ti_sigma", &multem::Input::obj_lens_ti_sigma)
    .def_readwrite("obj_lens_ti_beta", &multem::Input::obj_lens_ti_beta)
    .def_readwrite("obj_lens_ti_npts", &multem::Input::obj_lens_ti_npts)
    .def_readwrite("obj_lens_zero_defocus_type", &multem::Input::obj_lens_zero_defocus_type)
    .def_readwrite("obj_lens_zero_defocus_plane", &multem::Input::obj_lens_zero_defocus_plane)
    .def_readwrite("phase_shift", &multem::Input::phase_shift)
    .def_readwrite("detector", &multem::Input::detector)
    .def_readwrite("scanning_type", &multem::Input::scanning_type)
    .def_readwrite("scanning_square_pxs", &multem::Input::scanning_square_pxs)
    .def_readwrite("scanning_periodic", &multem::Input::scanning_periodic)
    .def_readwrite("scanning_ns", &multem::Input::scanning_ns)
    .def_readwrite("scanning_x0", &multem::Input::scanning_x0)
    .def_readwrite("scanning_y0", &multem::Input::scanning_y0)
    .def_readwrite("scanning_xe", &multem::Input::scanning_xe)
    .def_readwrite("scanning_ye", &multem::Input::scanning_ye)
    .def_readwrite("ped_nrot", &multem::Input::ped_nrot)
    .def_readwrite("ped_theta", &multem::Input::ped_theta)
    .def_readwrite("hci_nrot", &multem::Input::hci_nrot)
    .def_readwrite("hci_theta", &multem::Input::hci_theta)
    .def_readwrite("eels_Z", &multem::Input::eels_Z)
    .def_readwrite("eels_E_loss", &multem::Input::eels_E_loss)
    .def_readwrite("eels_collection_angle", &multem::Input::eels_collection_angle)
    .def_readwrite("eels_m_selection", &multem::Input::eels_m_selection)
    .def_readwrite("eels_channelling_type", &multem::Input::eels_channelling_type)
    .def_readwrite("eftem_Z", &multem::Input::eftem_Z)
    .def_readwrite("eftem_E_loss", &multem::Input::eftem_E_loss)
    .def_readwrite("eftem_collection_angle", &multem::Input::eftem_collection_angle)
    .def_readwrite("eftem_m_selection", &multem::Input::eftem_m_selection)
    .def_readwrite("eftem_channelling_type", &multem::Input::eftem_channelling_type)
    .def_readwrite("output_area_ix_0", &multem::Input::output_area_ix_0)
    .def_readwrite("output_area_iy_0", &multem::Input::output_area_iy_0)
    .def_readwrite("output_area_ix_e", &multem::Input::output_area_ix_e)
    .def_readwrite("output_area_iy_e", &multem::Input::output_area_iy_e)
    .def("asdict", &py::detail::Helpers<multem::Input>::asdict)
    .def(py::pickle(
        &py::detail::Helpers<multem::Input>::getstate,
        &py::detail::Helpers<multem::Input>::setstate))
    ;

  // Wrap the multem::SystemConfiguration class
  py::class_<multem::SystemConfiguration>(m, "SystemConfiguration")
    .def(py::init<>())
    .def_readwrite("device", &multem::SystemConfiguration::device)
    .def_readwrite("precision", &multem::SystemConfiguration::precision)
    .def_readwrite("cpu_ncores", &multem::SystemConfiguration::cpu_ncores)
    .def_readwrite("cpu_nthread", &multem::SystemConfiguration::cpu_nthread)
    .def_readwrite("gpu_device", &multem::SystemConfiguration::gpu_device)
    .def_readwrite("gpu_nstream", &multem::SystemConfiguration::gpu_nstream)
    .def(py::pickle(
        &py::detail::Helpers<multem::SystemConfiguration>::getstate,
        &py::detail::Helpers<multem::SystemConfiguration>::setstate))
    ;

  // Wrap a double and complex image class
  py::detail::image_class<double>(m, "ImageDouble");
  py::detail::image_class< std::complex<double> >(m, "ImageComplexDouble");

  // Wrap the multem::Data class
  py::class_<multem::Data>(m, "Data")
    .def(py::init<>())
    .def_readwrite("image_tot", &multem::Data::image_tot)
    .def_readwrite("image_coh", &multem::Data::image_coh)
    .def_readwrite("m2psi_tot", &multem::Data::m2psi_tot)
    .def_readwrite("m2psi_coh", &multem::Data::m2psi_coh)
    .def_readwrite("psi_coh", &multem::Data::psi_coh)
    .def_readwrite("V", &multem::Data::V)
    ;

  // Wrap the multem::Output class
  py::class_<multem::Output>(m, "Output")
    .def(py::init<>())
    .def_readwrite("dx", &multem::Output::dx)
    .def_readwrite("dy", &multem::Output::dy)
    .def_readwrite("x", &multem::Output::x)
    .def_readwrite("y", &multem::Output::y)
    .def_readwrite("thick", &multem::Output::thick)
    .def_readwrite("data", &multem::Output::data)
    .def("asdict", &py::detail::Helpers<multem::Output>::asdict)
    ;

  // Wrap the multem::Masker class
  py::class_<multem::Masker>(m, "Masker")
    .def(py::init<>())
    .def(py::init<std::size_t, std::size_t, double>())
    .def("xsize", &multem::Masker::xsize)
    .def("ysize", &multem::Masker::ysize)
    .def("pixel_size", &multem::Masker::pixel_size)
    .def("shape", &multem::Masker::shape)
    .def("xmin", &multem::Masker::xmin)
    .def("ymin", &multem::Masker::ymin)
    .def("zmin", &multem::Masker::zmin)
    .def("xmax", &multem::Masker::xmax)
    .def("ymax", &multem::Masker::ymax)
    .def("zmax", &multem::Masker::zmax)
    .def("rotation_origin", &multem::Masker::rotation_origin)
    .def("rotation_angle", &multem::Masker::rotation_angle)
    .def("translation", &multem::Masker::translation)
    .def("ice_parameters", &multem::Masker::ice_parameters)
    .def("set_image_size", &multem::Masker::set_image_size)
    .def("set_pixel_size", &multem::Masker::set_pixel_size)
    .def("set_cube", &multem::Masker::set_cube)
    .def("set_cuboid", &multem::Masker::set_cuboid)
    .def("set_cylinder", &multem::Masker::set_cylinder)
    .def("set_rotation", &multem::Masker::set_rotation)
    .def("set_translation", &multem::Masker::set_translation)
    .def("set_ice_parameters", &multem::Masker::set_ice_parameters)
    .def("compute", &pybind11::detail::Masker_compute)
    ;

  // Wrap the multem::IceParameters class
  py::class_<multem::IceParameters>(m, "IceParameters")
    .def(py::init<>())
    .def_readwrite("m1", &multem::IceParameters::m1)
    .def_readwrite("m2", &multem::IceParameters::m2)
    .def_readwrite("s1", &multem::IceParameters::s1)
    .def_readwrite("s2", &multem::IceParameters::s2)
    .def_readwrite("a1", &multem::IceParameters::a1)
    .def_readwrite("a2", &multem::IceParameters::a2)
    .def_readwrite("density", &multem::IceParameters::density)
    ;

  // Expose the simulation function
  //
  // Since this function takes a long time to run and pybind by default holds
  // the python GIL for all C++ functions, we should release the GIL in order
  // to avoid instability in software using python threads. This was observed to
  // be an issue with dask parallism on the cluster
  m.def(
      "simulate", 
      &multem::simulate, 
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "simulate", 
      &py::detail::simulate_slices, 
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "simulate", 
      &multem::simulate_with_ice_approximation, 
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "compute_projected_potential", 
      &multem::compute_projected_potential, 
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "compute_projected_potential", 
      &multem::compute_projected_potential_with_ice_approximation, 
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "compute_ctf", 
      &multem::compute_ctf, 
      py::call_guard<py::gil_scoped_release>());

  // Expose the GPU functions
  m.def("is_gpu_available", &multem::is_gpu_available);
  m.def("number_of_gpu_available", &multem::number_of_gpu_available);

  m.def("mrad_to_sigma", &multem::mrad_to_sigma);
  m.def("iehwgd_to_sigma", &multem::iehwgd_to_sigma);
  m.def("hwhm_to_sigma", &multem::hwhm_to_sigma);

  m.def("crystal_by_layers", &multem::crystal_by_layers);
  m.def("compute_V_params", &multem::compute_V_params);

  // Expose some tests
  m.def("test_ice_potential_approximation", &multem::test_ice_potential_approximation);
	m.def("test_masker", &multem::test_masker);
}

