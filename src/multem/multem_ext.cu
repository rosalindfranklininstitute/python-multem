/*
 *  multem_ext.cu
 *
 *  Copyright (C) 2019 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the GPLv3 license, a copy of 
 *  which is included in the root directory of this package.
 */

#include <sstream>
#include <iostream>
#include <stdexcept>
#include <map>
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <multem/multem_ext.h>
#include <types.cuh>
#include <input_multislice.cuh>
#include <device_functions.cuh>
#include <crystal_spec.hpp>
#include <multem.cu>

namespace multem {

  namespace detail {

    /**
     * Template type containing enum string info
     */
    template <typename T>
    struct EnumStrings {
      static std::string name();
      static std::map<T, std::string> items();
    };

    /**
     * Convert the enum into a string
     * @param value The enum value
     * @returns The enum string
     */
    template <
      typename T,
      typename std::enable_if<std::is_enum<T>::value>::type* = nullptr>
    std::string to_string(const T &value) {
      return EnumStrings<T>::items()[value];
    }

    /**
     * Produce an error message if the case of an unknown enum string
     * @param name The enum string
     * @returns The error message
     */
    template<
      typename T,
      typename std::enable_if<std::is_enum<T>::value>::type* = nullptr>
    std::string unknown_enum_string_message(const std::string &name) {
      std::stringstream msg;
      msg << "Unknown enum string for enum "
          << "\"" << EnumStrings<T>::name() << "\"."
          << " Got " << name << ", expected one of:\n";
      for (auto it : EnumStrings<T>::items()) {
        msg << "    " << it.second << "\n";
      }
      return msg.str();
    }

    /**
     * Convert the string into an enum value
     * @param name The enum string
     * @returns The enum value
     */
    template<
      typename T,
      typename std::enable_if<std::is_enum<T>::value>::type* = nullptr>
    T from_string(std::string name) {
      for (auto it : EnumStrings<T>::items()) {
        if (it.second == name) {
          return it.first;
        }
      }
      throw std::runtime_error(unknown_enum_string_message<T>(name));
    }

    /**
     * Specialization for mt::eDevice 
     */
    template <>
    struct EnumStrings<mt::eDevice> {
      
      static std::string name() {
        return "eDevice";
      }

      static std::map<mt::eDevice, std::string> items() {
        return {
          { mt::e_host, "host" },
          { mt::e_device, "device" },
          { mt::e_host_device, "host_device" },
        };
      }
    };

    /**
     * Specialization for mt::ePrecision 
     */
    template<>
    struct EnumStrings<mt::ePrecision> {
      
      static std::string name() {
        return "mt::ePrecision";
      }
      
      static std::map<mt::ePrecision, std::string> items() {
        return {
          { mt::eP_float, "float" },
          { mt::eP_double, "double" }, 
        };
      }
    };

    /**
     * Specialization for mt::eElec_Spec_Int_Model 
     */
    template<>
    struct EnumStrings<mt::eElec_Spec_Int_Model> {
      
      static std::string name() {
        return "mt::eElec_Spec_Int_Model";
      }
      
      static std::map<mt::eElec_Spec_Int_Model, std::string> items() {
        return {
          { mt::eESIM_Multislice, "Multislice" },
          { mt::eESIM_Phase_Object, "Phase_Object" },
          { mt::eESIM_Weak_Phase_Object, "Weak_Phase_Object" },
        };
      }
    };

    /**
     * Specialization for mt::ePotential_Type 
     */
    template<>
    struct EnumStrings<mt::ePotential_Type> {
      
      static std::string name() {
        return "mt::ePotential_Type";
      }
      
      static std::map<mt::ePotential_Type, std::string> items() {
        return {
          { mt::ePT_Doyle_0_4, "Doyle_0_4" },
          { mt::ePT_Peng_0_4, "Peng_0_4" },
          { mt::ePT_Peng_0_12, "Peng_0_12" },
          { mt::ePT_Kirkland_0_12, "Kirkland_0_12" },
          { mt::ePT_Weickenmeier_0_12, "Weickenmeier_0_12" },
          { mt::ePT_Lobato_0_12, "Lobato_0_12" },
          { mt::ePT_none, "none" },
        };
      }
    };

    /**
     * Specialization for mt::ePhonon_Model 
     */
    template<>
    struct EnumStrings<mt::ePhonon_Model> {
      
      static std::string name() {
        return "mt::ePhonon_Model";
      }
      
      static std::map<mt::ePhonon_Model, std::string> items() {
        return {
          { mt::ePM_Still_Atom, "Still_Atom" },
          { mt::ePM_Absorptive_Model, "Absorptive_Model" },
          { mt::ePM_Frozen_Phonon, "Frozen_Phonon" },
        };
      }
    };

    /**
     * Specialization for mt::eRot_Point_Type 
     */
    template<>
    struct EnumStrings<mt::eRot_Point_Type> {
      
      static std::string name() {
        return "mt::eRot_Point_Type";
      }
      
      static std::map<mt::eRot_Point_Type, std::string> items() {
        return {
          { mt::eRPT_geometric_center, "geometric_center" },
          { mt::eRPT_User_Define, "User_Define" },
        };
      }
    };

    /**
     * Specialization for mt::eThick_Type 
     */
    template<>
    struct EnumStrings<mt::eThick_Type> {
      
      static std::string name() {
        return "mt::eThick_Type";
      }
      
      static std::map<mt::eThick_Type, std::string> items() {
        return {
          { mt::eTT_Whole_Spec, "Whole_Spec" },
          { mt::eTT_Through_Thick, "Through_Thick" },
          { mt::eTT_Through_Slices, "Through_Slices" },
        };
      }
    };

    /**
     * Specialization for mt::ePotential_Slicing 
     */
    template<>
    struct EnumStrings<mt::ePotential_Slicing> {
      
      static std::string name() {
        return "mt::ePotential_Slicing";
      }
      
      static std::map<mt::ePotential_Slicing, std::string> items() {
        return {
          { mt::ePS_Planes, "Planes" },
          { mt::ePS_dz_Proj, "dz_Proj" },
          { mt::ePS_dz_Sub, "dz_Sub" },
          { mt::ePS_Auto, "Auto" },
        };
      }
    };

    /**
     * Specialization for mt::eTEM_Sim_Type 
     */
    template<>
    struct EnumStrings<mt::eTEM_Sim_Type> {
      
      static std::string name() {
        return "mt::eTEM_Sim_Type";
      }
      
      static std::map<mt::eTEM_Sim_Type, std::string> items() {
        return {
          { mt::eTEMST_STEM , "STEM" },
          { mt::eTEMST_ISTEM , "ISTEM" },
          { mt::eTEMST_CBED , "CBED" },
          { mt::eTEMST_CBEI , "CBEI" },
          { mt::eTEMST_ED , "ED" },
          { mt::eTEMST_HRTEM , "HRTEM" },
          { mt::eTEMST_PED , "PED" },
          { mt::eTEMST_HCTEM , "HCTEM" },
          { mt::eTEMST_EWFS , "EWFS" },
          { mt::eTEMST_EWRS , "EWRS" },
          { mt::eTEMST_EELS , "EELS" },
          { mt::eTEMST_EFTEM , "EFTEM" },
          { mt::eTEMST_IWFS , "IWFS" },
          { mt::eTEMST_IWRS , "IWRS" },
          { mt::eTEMST_PPFS , "PPFS" },
          { mt::eTEMST_PPRS , "PPRS" },
          { mt::eTEMST_TFFS , "TFFS" },
          { mt::eTEMST_TFRS , "TFRS" },
          { mt::eTEMST_PropFS , "PropFS" },
          { mt::eTEMST_PropRS , "PropRS" },
        };
      }
    };

    /**
     * Specialization for mt::eIncident_Wave_Type 
     */
    template<>
    struct EnumStrings<mt::eIncident_Wave_Type> {
      
      static std::string name() {
        return "mt::eIncident_Wave_Type";
      }
      
      static std::map<mt::eIncident_Wave_Type, std::string> items() {
        return {
          { mt::eIWT_Plane_Wave, "Plane_Wave" },
          { mt::eIWT_Convergent_Wave, "Convergent_Wave" },
          { mt::eIWT_User_Define_Wave, "User_Define_Wave" },
          { mt::eIWT_Auto, "Auto" },
        };
      }
    };

    /**
     * Specialization for mt::eIllumination_Model 
     */
    template<>
    struct EnumStrings<mt::eIllumination_Model> {
      
      static std::string name() {
        return "mt::eIllumination_Model";
      }
      
      static std::map<mt::eIllumination_Model, std::string> items() {
        return {
          { mt::eIM_Coherent, "Coherent" },
          { mt::eIM_Partial_Coherent, "Partial_Coherent" },
          { mt::eIM_Trans_Cross_Coef, "Trans_Cross_Coef" },
          { mt::eIM_Full_Integration, "Full_Integration" },
          { mt::eIM_none, "none" },
        };
      }
    };
      
    /**
     * Specialization for mt::eOperation_Mode 
     */
    template<>
    struct EnumStrings<mt::eOperation_Mode> {
      
      static std::string name() {
        return "mt::eOperation_Mode";
      }
      
      static std::map<mt::eOperation_Mode, std::string> items() {
        return {
          { mt::eOM_Normal, "Normal" },
          { mt::eOM_Advanced, "Advanced" },
        };
      }
    };

    /**
     * Specialization for mt::eLens_Var_Type 
     */
    template<>
    struct EnumStrings<mt::eLens_Var_Type> {
      
      static std::string name() {
        return "mt::eLens_Var_Type";
      }
      
      static std::map<mt::eLens_Var_Type, std::string> items() {
        return {
          { mt::eLVT_off, "off" },
          { mt::eLVT_m, "m" },
          { mt::eLVT_f, "f" },
          { mt::eLVT_Cs3, "Cs3" },
          { mt::eLVT_Cs5, "Cs5" },
          { mt::eLVT_mfa2, "mfa2" },
          { mt::eLVT_afa2, "afa2" },
          { mt::eLVT_mfa3, "mfa3" },
          { mt::eLVT_afa3, "afa3" },
          { mt::eLVT_inner_aper_ang, "inner_aper_ang" },
          { mt::eLVT_outer_aper_ang, "outer_aper_ang" },
        };
      }
    };

    /**
     * Specialization for mt::eTemporal_Spatial_Incoh 
     */
    template<>
    struct EnumStrings<mt::eTemporal_Spatial_Incoh> {
      
      static std::string name() {
        return "mt::eTemporal_Spatial_Incoh";
      }
      
      static std::map<mt::eTemporal_Spatial_Incoh, std::string> items() {
        return {
          { mt::eTSI_Temporal_Spatial, "Temporal_Spatial" },
          { mt::eTSI_Temporal, "Temporal" },
          { mt::eTSI_Spatial, "Spatial" },
          { mt::eTSI_none, "none" },
        };
      }
    };

    /**
     * Specialization for mt::eZero_Defocus_Type 
     */
    template<>
    struct EnumStrings<mt::eZero_Defocus_Type> {
      
      static std::string name() {
        return "mt::eZero_Defocus_Type";
      }
      
      static std::map<mt::eZero_Defocus_Type, std::string> items() {
        return {
          { mt::eZDT_First, "First" },
          { mt::eZDT_Middle, "Middle" },
          { mt::eZDT_Last, "Last" },
          { mt::eZDT_User_Define, "User_Define" },
        };
      }
    };

    /**
     * Specialization for mt::eScanning_Type 
     */
    template<>
    struct EnumStrings<mt::eScanning_Type> {
      
      static std::string name() {
        return "mt::eScanning_Type";
      }
      
      static std::map<mt::eScanning_Type, std::string> items() {
        return {
          { mt::eST_Line, "Line" },
          { mt::eST_Area, "Area" },
        };
      }
    };

    /**
     * Specialization for mt::eDetector_Type 
     */
    template<>
    struct EnumStrings<mt::eDetector_Type> {
      
      static std::string name() {
        return "mt::eDetector_Type";
      }
      
      static std::map<mt::eDetector_Type, std::string> items() {
        return {
          { mt::eDT_Circular, "Circular" },
          { mt::eDT_Radial, "Radial" },
          { mt::eDT_Matrix, "Matrix" },
        };
      }
    };

    /**
     * Specialization for mt::eChannelling_Type 
     */
    template<>
    struct EnumStrings<mt::eChannelling_Type> {
      
      static std::string name() {
        return "mt::eChannelling_Type";
      }
      
      static std::map<mt::eChannelling_Type, std::string> items() {
        return {
          { mt::eCT_Single_Channelling, "Single_Channelling" },
          { mt::eCT_Mixed_Channelling, "Mixed_Channelling" },
          { mt::eCT_Double_Channelling, "Double_Channelling" },
        };
      }
    };

    template <typename FloatType, mt::eDevice DeviceType>
    void run_multislice_internal(
      const mt::System_Configuration &system_conf,
      mt::Input_Multislice<FloatType> &input_multislice,
      mt::Output_Multislice<FloatType> &output_multislice) {
  
      // Set the system configration    
      input_multislice.system_conf = system_conf;

      // Open a stream
      mt::Stream<DeviceType> stream(system_conf.nstream);

      // Create the FFT object
      mt::FFT<FloatType, DeviceType> fft_2d;
      fft_2d.create_plan_2d(
        input_multislice.grid_2d.ny, 
        input_multislice.grid_2d.nx, 
        system_conf.nstream);

      // Setup the multislice simulation 
      mt::Multislice<FloatType, DeviceType> multislice;
      multislice.set_input_data(&input_multislice, &stream, &fft_2d);

      // Set the input data
      output_multislice.set_input_data(&input_multislice);

      // Perform the multislice simulation
      multislice(output_multislice);
      stream.synchronize();

      // Get the results
      output_multislice.gather();
      output_multislice.clean_temporal();
      fft_2d.cleanup();

      // If there was an error then throw and exception
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::stringstream message;
        message << "CUDA error: " << cudaGetErrorString(err) << "\n";
        throw std::runtime_error(message.str());
      }
    }
 
    mt::System_Configuration read_system_configuration(const SystemConfiguration &config) {
      mt::System_Configuration system_conf;
      system_conf.device = detail::from_string<mt::eDevice>(config.device);
      system_conf.precision = detail::from_string<mt::ePrecision>(config.precision);
      system_conf.cpu_ncores = config.cpu_ncores;
      system_conf.cpu_nthread = config.cpu_nthread;
      system_conf.gpu_device = config.gpu_device;
      system_conf.gpu_nstream = config.gpu_nstream;
      system_conf.active = true;
      system_conf.validate_parameters();
      system_conf.set_device();
      return system_conf;
    } 
    
    template <typename FloatType>
    mt::Input_Multislice<FloatType> read_input_multislice(
        const Input &input,
        bool full = true) {
      mt::Input_Multislice<FloatType> input_multislice;

      // Simulation type
      input_multislice.simulation_type = 
        detail::from_string<mt::eTEM_Sim_Type>(input.simulation_type);
      input_multislice.interaction_model = 
        detail::from_string<mt::eElec_Spec_Int_Model>(input.interaction_model);
      input_multislice.potential_type = 
        detail::from_string<mt::ePotential_Type>(input.potential_type);
      input_multislice.operation_mode = 
        detail::from_string<mt::eOperation_Mode>(input.operation_mode);
      input_multislice.reverse_multislice = input.reverse_multislice;

      // Electron-Phonon interaction model
      input_multislice.pn_model = detail::from_string<mt::ePhonon_Model>(input.pn_model);
      input_multislice.pn_coh_contrib = input.pn_coh_contrib;
      input_multislice.pn_single_conf = input.pn_single_conf;
      input_multislice.pn_nconf = input.pn_nconf;
      input_multislice.pn_dim.set(input.pn_dim);
      input_multislice.pn_seed = input.pn_seed;

      // Specimen
      bool pbc_xy = true;

      // Set the specimen
      if (input_multislice.is_specimen_required())
      {
        // Set the amorphous layer information
        mt::Vector<mt::Amorp_Lay_Info<FloatType>, mt::e_host> amorp_lay_info;
        for (auto item : input.spec_amorp) {
          mt::Amorp_Lay_Info<FloatType> value;
          value.z_0 = item.z_0;
          value.z_e = item.z_e;
          value.dz = item.dz;
          amorp_lay_info.push_back(value);
        }

        if (full) {
          input_multislice.atoms.set_crystal_parameters(
              input.spec_cryst_na, 
              input.spec_cryst_nb, 
              input.spec_cryst_nc, 
              input.spec_cryst_a, 
              input.spec_cryst_b, 
              input.spec_cryst_c, 
              input.spec_cryst_x0, 
              input.spec_cryst_y0);
          input_multislice.atoms.set_amorphous_parameters(amorp_lay_info);
          input_multislice.atoms.resize(input.spec_atoms.size());
          for(auto i = 0; i < input.spec_atoms.size(); ++i) {
            input_multislice.atoms.Z[i] = input.spec_atoms[i].element;
            input_multislice.atoms.x[i] = input.spec_atoms[i].x;
            input_multislice.atoms.y[i] = input.spec_atoms[i].y;
            input_multislice.atoms.z[i] = input.spec_atoms[i].z;
            input_multislice.atoms.sigma[i] = input.spec_atoms[i].sigma;
            input_multislice.atoms.occ[i] = input.spec_atoms[i].occupancy;
            input_multislice.atoms.region[i] = abs(input.spec_atoms[i].region);
            input_multislice.atoms.charge[i] = input.spec_atoms[i].charge;
          }
          input_multislice.atoms.get_statistic();
        }

        // Specimen rotation
        input_multislice.spec_rot_theta = input.spec_rot_theta*mt::c_deg_2_rad;
        input_multislice.spec_rot_u0 = mt::r3d<FloatType>(
            input.spec_rot_u0[0],
            input.spec_rot_u0[1],
            input.spec_rot_u0[2]);
        input_multislice.spec_rot_u0.normalized();
        input_multislice.spec_rot_center_type =
          detail::from_string<mt::eRot_Point_Type>(input.spec_rot_center_type);
        input_multislice.spec_rot_center_p = mt::r3d<FloatType>(
            input.spec_rot_center_p[0],
            input.spec_rot_center_p[1],
            input.spec_rot_center_p[2]);

        // Specimen thickness
        input_multislice.thick_type = detail::from_string<mt::eThick_Type>(input.thick_type);
        if (!input_multislice.is_whole_spec() && full) {
          input_multislice.thick.assign(input.thick.begin(), input.thick.end());
        }

        // Potential slicing
        input_multislice.potential_slicing =
          detail::from_string<mt::ePotential_Slicing>(input.potential_slicing);
      }

      // XY sampling
      auto nx = input.nx;
      auto ny = input.ny;
      bool bwl = input.bwl;
      input_multislice.grid_2d.set_input_data(
          nx, 
          ny, 
          input.spec_lx, 
          input.spec_ly, 
          input.spec_dz, 
          bwl, 
          pbc_xy);

      // Incident wave
      input_multislice.set_incident_wave_type(detail::from_string<mt::eIncident_Wave_Type>(input.iw_type));

      if (input_multislice.is_user_define_wave() && full) {
        input_multislice.iw_psi.assign(
            input.iw_psi.begin(),
            input.iw_psi.end());
      }

      // read iw_x and iw_y
      int n_iw_xy = std::min(input.iw_x.size(), input.iw_y.size());
      input_multislice.iw_x.assign(input.iw_x.begin(), input.iw_x.begin() + n_iw_xy);
      input_multislice.iw_y.assign(input.iw_y.begin(), input.iw_y.begin() + n_iw_xy);

      // Microscope parameter
      input_multislice.E_0 = input.E_0;
      input_multislice.theta = input.theta*mt::c_deg_2_rad;
      input_multislice.phi = input.phi*mt::c_deg_2_rad;

      // Illumination model
      input_multislice.illumination_model =
        detail::from_string<mt::eIllumination_Model>(input.illumination_model);
      input_multislice.temporal_spatial_incoh =
        detail::from_string<mt::eTemporal_Spatial_Incoh>(input.temporal_spatial_incoh);

      // Condenser lens
      input_multislice.cond_lens.m = input.cond_lens_m;
      input_multislice.cond_lens.c_10 = input.cond_lens_c_10;
      input_multislice.cond_lens.c_12 = input.cond_lens_c_12;
      input_multislice.cond_lens.phi_12 = input.cond_lens_phi_12*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_21 = input.cond_lens_c_21;
      input_multislice.cond_lens.phi_21 = input.cond_lens_phi_21*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_23 = input.cond_lens_c_23;
      input_multislice.cond_lens.phi_23 = input.cond_lens_phi_23*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_30 = input.cond_lens_c_30*mt::c_mm_2_Angs; 
      input_multislice.cond_lens.c_32 = input.cond_lens_c_32; 
      input_multislice.cond_lens.phi_32 = input.cond_lens_phi_32*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_34 = input.cond_lens_c_34;
      input_multislice.cond_lens.phi_34 = input.cond_lens_phi_34*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_41 = input.cond_lens_c_41;
      input_multislice.cond_lens.phi_41 = input.cond_lens_phi_41*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_43 = input.cond_lens_c_43;
      input_multislice.cond_lens.phi_43 = input.cond_lens_phi_43*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_45 = input.cond_lens_c_45;
      input_multislice.cond_lens.phi_45 = input.cond_lens_phi_45*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_50 = input.cond_lens_c_50*mt::c_mm_2_Angs;
      input_multislice.cond_lens.c_52 = input.cond_lens_c_52;
      input_multislice.cond_lens.phi_52 = input.cond_lens_phi_52*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_54 = input.cond_lens_c_54;
      input_multislice.cond_lens.phi_54 = input.cond_lens_phi_54*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_56 = input.cond_lens_c_56;
      input_multislice.cond_lens.phi_56 = input.cond_lens_phi_56*mt::c_deg_2_rad;
      input_multislice.cond_lens.inner_aper_ang = input.cond_lens_inner_aper_ang*mt::c_mrad_2_rad;
      input_multislice.cond_lens.outer_aper_ang = input.cond_lens_outer_aper_ang*mt::c_mrad_2_rad;

      // defocus spread function
      input_multislice.cond_lens.dsf_sigma = input.cond_lens_dsf_sigma;
      input_multislice.cond_lens.dsf_npoints = input.cond_lens_dsf_npoints;

      // source spread function
      input_multislice.cond_lens.ssf_sigma = input.cond_lens_ssf_sigma;
      input_multislice.cond_lens.ssf_npoints = input.cond_lens_ssf_npoints;

      // zero defocus reference
      input_multislice.cond_lens.zero_defocus_type = 
        detail::from_string<mt::eZero_Defocus_Type>(input.cond_lens_zero_defocus_type);
      input_multislice.cond_lens.zero_defocus_plane = input.cond_lens_zero_defocus_plane;
      input_multislice.cond_lens.set_input_data(input_multislice.E_0, input_multislice.grid_2d);

      // Objective lens
      input_multislice.obj_lens.m = input.obj_lens_m;
      input_multislice.obj_lens.c_10 = input.obj_lens_c_10;
      input_multislice.obj_lens.c_12 = input.obj_lens_c_12;
      input_multislice.obj_lens.phi_12 = input.obj_lens_phi_12*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_21 = input.obj_lens_c_21;
      input_multislice.obj_lens.phi_21 = input.obj_lens_phi_21*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_23 = input.obj_lens_c_23;
      input_multislice.obj_lens.phi_23 = input.obj_lens_phi_23*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_30 = input.obj_lens_c_30*mt::c_mm_2_Angs;
      input_multislice.obj_lens.c_32 = input.obj_lens_c_32;
      input_multislice.obj_lens.phi_32 = input.obj_lens_phi_32*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_34 = input.obj_lens_c_34;
      input_multislice.obj_lens.phi_34 = input.obj_lens_phi_34*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_41 = input.obj_lens_c_41;
      input_multislice.obj_lens.phi_41 = input.obj_lens_phi_41*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_43 = input.obj_lens_c_43;
      input_multislice.obj_lens.phi_43 = input.obj_lens_phi_43*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_45 = input.obj_lens_c_45;
      input_multislice.obj_lens.phi_45 = input.obj_lens_phi_45*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_50 = input.obj_lens_c_50*mt::c_mm_2_Angs;
      input_multislice.obj_lens.c_52 = input.obj_lens_c_52;
      input_multislice.obj_lens.phi_52 = input.obj_lens_phi_52*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_54 = input.obj_lens_c_54;
      input_multislice.obj_lens.phi_54 = input.obj_lens_phi_54*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_56 = input.obj_lens_c_56;
      input_multislice.obj_lens.phi_56 = input.obj_lens_phi_56*mt::c_deg_2_rad;
      input_multislice.obj_lens.inner_aper_ang = input.obj_lens_inner_aper_ang*mt::c_mrad_2_rad;
      input_multislice.obj_lens.outer_aper_ang = input.obj_lens_outer_aper_ang*mt::c_mrad_2_rad;

      // defocus spread function
      input_multislice.obj_lens.dsf_sigma = input.obj_lens_dsf_sigma;
      input_multislice.obj_lens.dsf_npoints = input.obj_lens_dsf_npoints;

      // source spread function
      input_multislice.obj_lens.ssf_sigma = input_multislice.cond_lens.ssf_sigma;
      input_multislice.obj_lens.ssf_npoints = input_multislice.cond_lens.ssf_npoints;

      // zero defocus reference
      input_multislice.obj_lens.zero_defocus_type = 
        detail::from_string<mt::eZero_Defocus_Type>(input.obj_lens_zero_defocus_type);
      input_multislice.obj_lens.zero_defocus_plane = input.obj_lens_zero_defocus_plane;
      input_multislice.obj_lens.set_input_data(input_multislice.E_0, input_multislice.grid_2d);

      // ISTEM/STEM 
      if (input_multislice.is_scanning()) {
        input_multislice.scanning.type = detail::from_string<mt::eScanning_Type>(input.scanning_type);
        input_multislice.scanning.pbc = input.scanning_periodic;
        input_multislice.scanning.ns = input.scanning_ns;
        input_multislice.scanning.x0 = input.scanning_x0;
        input_multislice.scanning.y0 = input.scanning_y0;
        input_multislice.scanning.xe = input.scanning_xe;
        input_multislice.scanning.ye = input.scanning_ye;
        input_multislice.scanning.set_grid();
      }

      if (input_multislice.is_STEM()) {
        FloatType lambda = mt::get_lambda(input_multislice.E_0);
        input_multislice.detector.type = detail::from_string<mt::eDetector_Type>(input.detector.type);

        switch (input_multislice.detector.type) {
        case mt::eDT_Circular: 
        {
          int ndetector = input.detector.cir.size();
          if (ndetector > 0) {
            input_multislice.detector.resize(ndetector);
            for (auto i = 0; i < input_multislice.detector.size(); i++) {
              auto inner_ang = input.detector.cir[i].inner_ang*mt::c_mrad_2_rad;
              auto outer_ang = input.detector.cir[i].outer_ang*mt::c_mrad_2_rad;
              input_multislice.detector.g_inner[i] = std::sin(inner_ang) / lambda;
              input_multislice.detector.g_outer[i] = std::sin(outer_ang) / lambda;
            }
          }
        }
        break;
        case mt::eDT_Radial:
        {
          int ndetector = input.detector.radial.size();
          if (ndetector > 0) {
            input_multislice.detector.resize(ndetector);
            for (auto i = 0; i < input_multislice.detector.size(); i++) {
              input_multislice.detector.fx[i].assign(
                  input.detector.radial[i].fx.begin(),
                  input.detector.radial[i].fx.end());
            }
          }
        }
        break;
        case mt::eDT_Matrix:
        {
          int ndetector = input.detector.matrix.size();
          if (ndetector > 0) {
            input_multislice.detector.resize(ndetector);
            for (auto i = 0; i < input_multislice.detector.size(); i++) {
              input_multislice.detector.fR[i].assign(
                  input.detector.matrix[i].fR.begin(),
                  input.detector.matrix[i].fR.end());
            }
          }
        }
        break;
        };
      } else if (input_multislice.is_PED()) {
        input_multislice.theta = input.ped_theta*mt::c_deg_2_rad;
        input_multislice.nrot = input.ped_nrot;
      } else if (input_multislice.is_HCTEM()) {
        input_multislice.theta = input.hci_theta*mt::c_deg_2_rad;
        input_multislice.nrot = input.hci_nrot;
      } else if (input_multislice.is_EELS()) {
        input_multislice.eels_fr.set_input_data(
          mt::eS_Reciprocal, 
          input_multislice.E_0, 
          input.eels_E_loss * mt::c_eV_2_keV, 
          input.eels_m_selection, 
          input.eels_collection_angle * mt::c_mrad_2_rad, 
          detail::from_string<mt::eChannelling_Type>(input.eels_channelling_type),
          input.eels_Z);
      } else if (input_multislice.is_EFTEM()) {
        input_multislice.eels_fr.set_input_data(
            mt::eS_Real, 
            input_multislice.E_0, 
            input.eftem_E_loss * mt::c_eV_2_keV, 
            input.eftem_m_selection, 
            input.eftem_collection_angle * mt::c_mrad_2_rad, 
            detail::from_string<mt::eChannelling_Type>(input.eftem_channelling_type),
            input.eftem_Z);
      }

      // Select the output region
      input_multislice.output_area.ix_0 = input.output_area_ix_0-1;
      input_multislice.output_area.iy_0 = input.output_area_iy_0-1;
      input_multislice.output_area.ix_e = input.output_area_ix_e-1;
      input_multislice.output_area.iy_e = input.output_area_iy_e-1;

      // Validate the input parameters
      input_multislice.validate_parameters();
      return input_multislice;
    }

    template <typename FloatType>
    Output write_output_multislice(const mt::Output_Multislice<FloatType> &output_multislice) {
      
      // Set some general properties
      Output result;
      result.dx = output_multislice.dx;
      result.dy = output_multislice.dy;
      result.x.assign(output_multislice.x.begin(), output_multislice.x.end());
      result.y.assign(output_multislice.y.begin(), output_multislice.y.end());
      result.thick.assign(output_multislice.thick.begin(), output_multislice.thick.end());
      result.data.resize(output_multislice.thick.size());

      // Write the output data
      if (output_multislice.is_STEM() || output_multislice.is_EELS()) {
        std::size_t nx = (output_multislice.scanning.is_line()) ? 1 : output_multislice.nx;
        std::size_t ny = output_multislice.ny;
        for (auto i = 0; i < output_multislice.thick.size(); ++i) {
          result.data[i].image_tot.resize(output_multislice.ndetector);
          if (output_multislice.pn_coh_contrib) {
            result.data[i].image_coh.resize(output_multislice.ndetector);
          }
          for (auto j = 0; j < output_multislice.ndetector; ++j) {
            result.data[i].image_tot[j] = Image<double>(
                output_multislice.image_tot[i].image[j].data(), 
                  Image<double>::shape_type({ ny, nx }));
            if (output_multislice.pn_coh_contrib) {
              result.data[i].image_coh[j] = Image<double>(
                  output_multislice.image_coh[i].image[j].data(), 
                    Image<double>::shape_type({ ny, nx }));
            }
          }
        }
      } else if (output_multislice.is_EWFS_EWRS()) {
        for (auto i = 0; i < output_multislice.thick.size(); ++i) {
          if (!output_multislice.is_EWFS_EWRS_SC()) {
            result.data[i].m2psi_tot = Image<double>(
                output_multislice.m2psi_tot[i].data(), 
                  Image<double>::shape_type({
                    (std::size_t) output_multislice.ny,
                    (std::size_t) output_multislice.nx}));
          }
          result.data[i].psi_coh = Image< std::complex<double> >(
              output_multislice.psi_coh[i].data(), 
              Image< std::complex<double> >::shape_type({
                (std::size_t) output_multislice.ny,
                (std::size_t) output_multislice.nx}));
        }
      } else {
        for (auto i = 0; i < output_multislice.thick.size(); ++i) {
          result.data[i].m2psi_tot = Image<double>(
              output_multislice.m2psi_tot[i].data(), 
                Image<double>::shape_type({
                  (std::size_t) output_multislice.ny,
                  (std::size_t) output_multislice.nx}));
          if (output_multislice.pn_coh_contrib) {
            result.data[i].m2psi_coh = Image<double>(
                output_multislice.m2psi_coh[i].data(), 
                Image<double>::shape_type({
                  (std::size_t) output_multislice.ny,
                  (std::size_t) output_multislice.nx}));
          }
        }
      }


      // Return the result
      return result;
    }
  }

  template <typename FloatType, mt::eDevice DeviceType>
  Output run_multislice(SystemConfiguration config, Input input) {

    // Initialise the system configuration and input structures 
    auto system_conf = detail::read_system_configuration(config);
    auto input_multislice = detail::read_input_multislice<FloatType>(input);
    input_multislice.system_conf = system_conf;

    // Create the output structure
    mt::Output_Multislice<FloatType> output_multislice;
   
    // Run the simulation 
    detail::run_multislice_internal<FloatType, DeviceType>(
      system_conf, input_multislice, output_multislice);

    // Return the output struct
    return detail::write_output_multislice(output_multislice);
  }

  Output simulate(SystemConfiguration config, Input input) {
    Output result;
    if (config.device == "host" && config.precision == "float") {
      result = run_multislice<float, mt::e_host>(config, input);
    } else if (config.device == "host" && config.precision == "double") {
      result = run_multislice<double, mt::e_host>(config, input);
    } else if (config.device == "device" && config.precision == "float") {
      result = run_multislice<float, mt::e_device>(config, input);
    } else if (config.device == "device" && config.precision == "double") {
      result = run_multislice<double, mt::e_device>(config, input);
    } else {
      if (config.device != "host" && config.device != "device") {
        throw std::runtime_error("Unknown device");
      }
      if (config.precision != "float" && config.precision != "double") {
        throw std::runtime_error("Unknown precision");
      }
    } 
    return result;
  }

  bool is_gpu_available() {
    return mt::is_gpu_available();
  }

  int number_of_gpu_available() {
    return mt::number_of_gpu_available();
  }
  
  double mrad_to_sigma(double E0, double theta) {
    return mt::rad_2_sigma(E0, theta * mt::c_mrad_2_rad);
  }

  double iehwgd_to_sigma(double value) {
    return mt::iehwgd_2_sigma(value);
  }

  std::vector<Atom> crystal_by_layers(const CrystalParameters &params) {
    
    // Get the layer data 
    mt::Vector<mt::Atom_Data<double>, mt::e_host> layers(params.layers.size());
    for(auto i = 0; i < params.layers.size(); ++i) {
      layers[i].resize(params.layers[i].size());
      for (auto j = 0; j < params.layers[i].size(); ++j) {
        layers[i].Z[j] = params.layers[i][j].element;
        layers[i].x[j] = params.layers[i][j].x;
        layers[i].y[j] = params.layers[i][j].y;
        layers[i].z[j] = params.layers[i][j].z;
        layers[i].sigma[j] = params.layers[i][j].sigma;
        layers[i].occ[j] = params.layers[i][j].occupancy;
        layers[i].region[j] = abs(params.layers[i][j].region);
        layers[i].charge[j] = params.layers[i][j].charge;
      }
    }

    // Get the atoms from the crystal specification
    mt::Crystal_Spec<double> crystal_spec;
    mt::Atom_Data<double> atoms;
    crystal_spec(
      params.na, 
      params.nb, 
      params.nc, 
      params.a, 
      params.b, 
      params.c, 
      layers, 
      atoms);

    // Copy to vector for output
    std::vector<Atom> result(atoms.size());
    for (auto i = 0; i < atoms.size(); ++i) {
      result[i].element = atoms.Z[i];
      result[i].x = atoms.x[i];
      result[i].y = atoms.y[i];
      result[i].z = atoms.z[i];
      result[i].sigma = atoms.sigma[i];
      result[i].occupancy = atoms.occ[i];
      result[i].region = atoms.region[i];
      result[i].charge = atoms.charge[i];
    }
    return result;
  }

}

