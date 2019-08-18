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
#include <multem/multem_ext.h>

namespace py = pybind11;

namespace pybind11 { namespace detail {
  
  template <> 
  class type_caster<multem::Atom> {
  public:
  
    PYBIND11_TYPE_CASTER(multem::Atom, _("multem::Atom"));

    bool load(object src, bool convert) {
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
  
  template <> 
  class type_caster<multem::AmorphousLayer> {
  public:
  
    PYBIND11_TYPE_CASTER(multem::AmorphousLayer, _("multem::AmorphousLayer"));

    bool load(object src, bool convert) {
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
 


 
  /* template <> */ 
  /* class type_caster<multem::STEMDetector::Angles> { */
  /* public: */
  
  /*   PYBIND11_TYPE_CASTER(multem::STEMDetector::Angles, _("multem::STEMDetector::Angles")); */

  /*   bool load(object src, bool convert) { */
  /*     if (py::isinstance<py::tuple>(src)) { */
  /*       py::tuple t = py::cast<py::tuple>(src); */
  /*       if (py::len(t) == 2) { */
  /*         value.inner_ang = py::cast<double>(t[0]); */
  /*         value.outer_ang = py::cast<double>(t[1]); */
  /*         return true; */
  /*       } */
  /*     } */
  /*     return false; */
  /*   } */

  /*   static handle cast(multem::STEMDetector::Angles src, return_value_policy policy, handle parent) { */
  /*     return py::make_tuple( */
  /*       src.inner_ang, */ 
  /*       src.outer_ang).release(); */
  /*   } */
  /* }; */
  
  /* template <> */ 
  /* class type_caster<multem::STEMDetector::Radial> { */
  /* public: */
  
  /*   PYBIND11_TYPE_CASTER(multem::STEMDetector::Radial, _("multem::STEMDetector::Radial")); */

  /*   bool load(object src, bool convert) { */
  /*     if (py::isinstance<py::tuple>(src)) { */
  /*       py::tuple t = py::cast<py::tuple>(src); */
  /*       if (py::len(t) == 2) { */
  /*         value.x = py::cast<double>(t[0]); */
  /*         value.fx = py::cast<double>(t[1]); */
  /*         return true; */
  /*       } */
  /*     } */
  /*     return false; */
  /*   } */

  /*   static handle cast(multem::STEMDetector::Radial src, return_value_policy policy, handle parent) { */
  /*     return py::make_tuple( */
  /*       src.x, */ 
  /*       src.fx).release(); */
  /*   } */
  /* }; */
  
  /* template <> */ 
  /* class type_caster<multem::STEMDetector::Matrix> { */
  /* public: */
  
  /*   PYBIND11_TYPE_CASTER(multem::STEMDetector::Matrix, _("multem::STEMDetector::Matrix")); */

  /*   bool load(object src, bool convert) { */
  /*     if (py::isinstance<py::tuple>(src)) { */
  /*       py::tuple t = py::cast<py::tuple>(src); */
  /*       if (py::len(t) == 2) { */
  /*         value.R = py::cast<double>(t[0]); */
  /*         value.fR = py::cast<double>(t[1]); */
  /*         return true; */
  /*       } */
  /*     } */
  /*     return false; */
  /*   } */

  /*   static handle cast(multem::STEMDetector::Matrix src, return_value_policy policy, handle parent) { */
  /*     return py::make_tuple( */
  /*       src.R, */ 
  /*       src.fR).release(); */
  /*   } */
  /* }; */

  template <typename T>
  py::class_< multem::Image<T> > image_class(py::module &m, const char *name) {
    return py::class_< multem::Image<T> >(m, name, py::buffer_protocol())
      .def(py::init<>())
      .def(py::init([](py::array_t<T> array) -> multem::Image<T> {
        py::buffer_info buffer = array.request();
        if (buffer.ndim != 2) {
          throw std::runtime_error("Number of dimensions must be two");
        }
        return multem::Image<T>(
          (T *) buffer.ptr, 
          typename multem::Image<T>::shape_type({
            buffer.shape[0], 
            buffer.shape[1]}));
      }))
      .def_buffer([](multem::Image<T>& self) -> pybind11::buffer_info { 
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
      });
  }
}}


PYBIND11_MODULE(multem_ext, m)
{
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
 
  py::class_<multem::STEMDetector>(m, "STEMDetector")
    .def_readwrite("type", &multem::STEMDetector::type)
    .def_readwrite("cir", &multem::STEMDetector::cir)
    .def_readwrite("radial", &multem::STEMDetector::radial)
    .def_readwrite("matrix", &multem::STEMDetector::matrix)
    ;

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
    .def_readwrite("cond_lens_ssf_sigma", &multem::Input::cond_lens_ssf_sigma)
    .def_readwrite("cond_lens_ssf_npoints", &multem::Input::cond_lens_ssf_npoints)
    .def_readwrite("cond_lens_dsf_sigma", &multem::Input::cond_lens_dsf_sigma)
    .def_readwrite("cond_lens_dsf_npoints", &multem::Input::cond_lens_dsf_npoints)
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
    .def_readwrite("obj_lens_dsf_sigma", &multem::Input::obj_lens_dsf_sigma)
    .def_readwrite("obj_lens_dsf_npoints", &multem::Input::obj_lens_dsf_npoints)
    .def_readwrite("obj_lens_zero_defocus_type", &multem::Input::obj_lens_zero_defocus_type)
    .def_readwrite("obj_lens_zero_defocus_plane", &multem::Input::obj_lens_zero_defocus_plane)
    .def_readwrite("detector", &multem::Input::detector)
    .def_readwrite("scanning_type", &multem::Input::scanning_type)
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
    ;

  py::class_<multem::SystemConfiguration>(m, "SystemConfiguration")
    .def(py::init<>())
    .def_readwrite("device", &multem::SystemConfiguration::device)
    .def_readwrite("precision", &multem::SystemConfiguration::precision)
    .def_readwrite("cpu_ncores", &multem::SystemConfiguration::cpu_ncores)
    .def_readwrite("cpu_nthread", &multem::SystemConfiguration::cpu_nthread)
    .def_readwrite("gpu_device", &multem::SystemConfiguration::gpu_device)
    .def_readwrite("gpu_nstream", &multem::SystemConfiguration::gpu_nstream)
    ;

  py::detail::image_class<double>(m, "ImageDouble");
  py::detail::image_class< std::complex<double> >(m, "ImageComplexDouble");

  py::class_<multem::Data>(m, "Data")
    .def(py::init<>())
    .def_readwrite("image_tot", &multem::Data::image_tot)
    .def_readwrite("image_coh", &multem::Data::image_coh)
    .def_readwrite("m2psi_tot", &multem::Data::m2psi_tot)
    .def_readwrite("m2psi_coh", &multem::Data::m2psi_coh)
    .def_readwrite("psi_coh", &multem::Data::psi_coh)
    ;

  py::class_<multem::Output>(m, "Output")
    .def(py::init<>())
    .def_readwrite("dx", &multem::Output::dx)
    .def_readwrite("dy", &multem::Output::dy)
    .def_readwrite("x", &multem::Output::x)
    .def_readwrite("y", &multem::Output::y)
    .def_readwrite("thick", &multem::Output::thick)
    .def_readwrite("data", &multem::Output::data)
    ;

  m.def("simulate", &multem::simulate);

  m.def("is_gpu_available", &multem::is_gpu_available);
  m.def("number_of_gpu_available", &multem::number_of_gpu_available);

  m.def("mrad_to_sigma", &multem::mrad_to_sigma);
  m.def("iehwgd_to_sigma", &multem::iehwgd_to_sigma);

  m.def("crystal_by_layers", &multem::crystal_by_layers);
}
