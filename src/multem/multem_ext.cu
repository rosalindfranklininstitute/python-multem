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
#include <boost/math/special_functions/gamma.hpp>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <types.cuh>
#include <input_multislice.cuh>
#include <device_functions.cuh>
#include <crystal_spec.hpp>
#include <input_multislice.cuh>
#include <output_multislice.hpp>
#include <multislice.cuh>
#include <multem/multem_ext.h>

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
      throw multem::Error(unknown_enum_string_message<T>(name));
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


    /**
     * Compute the FT of the Gaussian Random Field
     */
    template <typename Generator, typename T>
    struct ComputeGaussianRandomField {

      Generator gen;
      double a1;
      double a2;
      double a3;
      double m1;
      double m2;
      double m3;
      double s1;
      double s2;
      double s3;
      double xsize;
      double ysize;

      /**
       * Initialise
       */
      ComputeGaussianRandomField(
            const Generator &gen_, 
            double a1_,
            double a2_,
            double a3_,
            double m1_,
            double m2_,
            double m3_,
            double s1_,
            double s2_,
            double s3_,
            size_t xsize_, 
            size_t ysize_)
        : gen(gen_),
          a1(a1_),
          a2(a2_),
          a3(a3_),
          m1(m1_),
          m2(m2_),
          m3(m3_),
          s1(s1_),
          s2(s2_),
          s3(s3_),
          xsize(xsize_),
          ysize(ysize_) {}

      /**
       * Compute the FT of the GRF at this index
       */
      DEVICE_CALLABLE
      T operator()(size_t index) const {
        size_t j = index / xsize;
        size_t i = index - j * xsize;

        // The uniform distribution
        thrust::uniform_real_distribution<double> uniform(0, 2*M_PI);

        // Initialise the random number generator
        Generator rnd = gen;
        rnd.discard(index);

        // Compute the power spectrum and phase
        double xd = (i-xsize/2.0) / xsize;
        double yd = (j-ysize/2.0) / ysize;
        double r = sqrt(xd*xd+yd*yd);
        double power = 
          a1 * (1.0/sqrt(2*M_PI*s1*s1)) * exp(-0.5*(r-m1)*(r-m1)/(s1*s1)) +
          a2 * (1.0/sqrt(2*M_PI*s2*s2)) * exp(-0.5*(r-m2)*(r-m2)/(s2*s2)) +
          a3 * (1.0/sqrt(2*M_PI*s3*s3)) * exp(-0.5*(r-m3)*(r-m3)/(s3*s3));
        double amplitude = sqrt(power);
        double phase = uniform(rnd);
        return (T)(amplitude) * exp(T(0, phase)); 
      }
    };

    struct Normalize {
    
      const double mean;
      const double sdev;

      Normalize(double m, double s):
        mean(m),
        sdev(s) {}

      template <typename T>
      DEVICE_CALLABLE
      T operator()(const T x) const {
        return (x - mean) / sdev;
      }
    };

    /**
     * Convert the Gaussian variable into a Gamma variable
     */
    struct GaussianToGamma {

      const double *gx;
      const double *gy;
      size_t size;

      /**
       * Initialise
       */
      GaussianToGamma(const double *gx_, const double *gy_, size_t size_)
        : gx(gx_),
          gy(gy_),
          size(size_) {
        MULTEM_ASSERT(gx != NULL && gy != NULL && size > 0);    
      }

      /**
       * Convert Gaussian to Uniform and Uniform to Gamma using
       * the Gaussian CDF and Gamma Quantile function lookup
       */
      DEVICE_CALLABLE
      double operator()(double x, bool m) const {
        double c = 0.5 * (1 + erf(x/sqrt(2.0f)));
        size_t i = (size_t)min(max(floor(c * size),(double)0), (double)(size-2));
        double gx0 = gx[i];
        double gx1 = gx[i+1];
        double gy0 = gy[i];
        double gy1 = gy[i+1];
        double g = gy0 + (gy1 - gy0) / (gx1 - gx0) * (c - gx0);
        return min(g, 70.0) * m;
      }

    };


    /**
     * Compute an approximation of the ice potential in a slice by modelling as
     * a Gamma distribution with a certain power spectrum
     */
    template <typename FloatType, mt::eDevice DeviceType>
    class IcePotentialApproximation : public mt::PotentialFunction<FloatType, DeviceType> {
    public:
			
      using T_r = FloatType;
			using T_c = complex<FloatType>;

      thrust::default_random_engine gen_;
      double alpha_a1_;
      double alpha_a2_;
      double alpha_m1_;
      double alpha_m2_;
      double alpha_s1_;
      double alpha_s2_;
      double theta_a1_;
      double theta_a2_;
      double theta_m1_;
      double theta_m2_;
      double theta_s1_;
      double theta_s2_;
      double a1_;
      double a2_;
      double a3_;
      double m1_;
      double m2_;
      double m3_;
      double s1_;
      double s2_;
      double s3_;
      mt::Grid_2d<FloatType> grid_2d_;
      mt::Vector<T_c, DeviceType> fft_data_;
      bool fft_data_use_real_;
      mt::Vector<T_r, DeviceType> random_field_;
      mt::Vector<bool, DeviceType> mask_;
      mt::FFT<T_r, DeviceType> *fft_2d_;
      Masker masker_;
      
      /**
       * Initialise
       */
      IcePotentialApproximation()
        : gen_(std::random_device()()),
          alpha_m1_(5.482149),
          alpha_m2_(11.814987),
          alpha_s1_(2.272096),
          alpha_s2_(3.644394),
          alpha_a1_(3.615568),
          alpha_a2_(23.229554),
          theta_m1_(0.604256),
          theta_m2_(-10.000000),
          theta_s1_(1.654867),
          theta_s2_(35.495530),
          theta_a1_(41.759711),
          theta_a2_(1077.047912),
          m1_(0.310186),
          m2_(0.315253),
          m3_(0.018720),
          s1_(0.041496),
          s2_(0.091280),
          s3_(0.244653),
          a1_(0.048754),
          a2_(0.314249),
          a3_(0.456534),
          /* m1_(0.310692), */
          /* m2_(0.315979), */
          /* m3_(0.000000), */
          /* s1_(0.040477), */
          /* s2_(0.086878), */
          /* s3_(0.332666), */
          /* a1_(0.037120), */
          /* a2_(0.240276), */
          /* a3_(0.715884), */
          /* m1_(1.712697), */
          /* m2_(0.099320), */
          /* m3_(0.307748), */
          /* s1_(18.959692), */
          /* s2_(0.344997), */
          /* s3_(0.069793), */
          /* a1_(0.000009), */
          /* a2_(0.709259), */
          /* a3_(0.224001), */
          fft_data_use_real_(true),
          fft_2d_(NULL) {}

      /**
       * Set the random engine
       */
      void set_random_engine(const thrust::default_random_engine &gen) {
        gen_ = gen;
      }

      /**
       * Get the masker
       */
      const Masker& masker() const {
        return masker_;
      }

      /**
       * Get the masker
       */
      Masker& masker() {
        return masker_;
      }

      /**
       * Set the masker
       */
      void set_masker(const Masker &masker) {
        masker_ = masker;
      }

      /**
       * Set the power spectrum parameters
       */
      void set_power_spectrum_parameters(
          double a1,
          double a2,
          double a3,
          double m1,
          double m2,
          double m3,
          double s1,
          double s2,
          double s3) {
        MULTEM_ASSERT(a1_ >= 0);
        MULTEM_ASSERT(a2_ >= 0);
        MULTEM_ASSERT(a3_ >= 0);
        MULTEM_ASSERT(m1_ >= 0);
        MULTEM_ASSERT(m2_ >= 0);
        MULTEM_ASSERT(m3_ >= 0);
        MULTEM_ASSERT(s1_ >= 0);
        MULTEM_ASSERT(s2_ >= 0);
        MULTEM_ASSERT(s3_ >= 0);
        a1_ = a1;
        a2_ = a2;
        a3_ = a3;
        m1_ = m1;
        m2_ = m2;
        m3_ = m3;
        s1_ = s1;
        s2_ = s2;
        s3_ = s3;
        fft_data_use_real_ = true;
      }
      
      /**
       * Set the gamma parameters
       */
      void set_gamma_parameters(
          double alpha_a1,
          double alpha_a2,
          double alpha_m1,
          double alpha_m2,
          double alpha_s1,
          double alpha_s2,
          double theta_a1,
          double theta_a2,
          double theta_m1,
          double theta_m2,
          double theta_s1,
          double theta_s2) {
        MULTEM_ASSERT(alpha_a1 >= 0);
        MULTEM_ASSERT(alpha_a2 >= 0);
        MULTEM_ASSERT(alpha_m1 >= 0);
        MULTEM_ASSERT(alpha_m2 >= 0);
        MULTEM_ASSERT(alpha_s1 >= 0);
        MULTEM_ASSERT(alpha_s2 >= 0);
        MULTEM_ASSERT(theta_a1 >= 0);
        MULTEM_ASSERT(theta_a2 >= 0);
        MULTEM_ASSERT(theta_m1 >= 0);
        MULTEM_ASSERT(theta_m2 >= 0);
        MULTEM_ASSERT(theta_s1 >= 0);
        MULTEM_ASSERT(theta_s2 >= 0);
        alpha_a1_ = alpha_a1;
        alpha_a2_ = alpha_a2;
        alpha_m1_ = alpha_m1;
        alpha_m2_ = alpha_m2;
        alpha_s1_ = alpha_s1;
        alpha_s2_ = alpha_s2;
        theta_a1_ = theta_a1;
        theta_a2_ = theta_a2;
        theta_m1_ = theta_m1;
        theta_m2_ = theta_m2;
        theta_s1_ = theta_s1;
        theta_s2_ = theta_s2;
      }

      /**
       * Set the grid size
       */
      void set_grid(mt::Grid_2d<FloatType> grid_2d) {

        // If we set the grid then resize everything
        if (grid_2d.nx != grid_2d_.nx || grid_2d.ny != grid_2d_.ny) {
          grid_2d_ = grid_2d,
          fft_data_.resize(grid_2d.nx * grid_2d.ny);
          random_field_.resize(grid_2d.nx * grid_2d.ny);
          mask_.resize(grid_2d.nx * grid_2d.ny);
          fft_data_use_real_ = true;
        }
      }
      
      /**
       * Set the FFT instance
       */
      void set_fft_2d(mt::FFT<T_r, DeviceType> *fft_2d) {
        fft_2d_ = fft_2d;
      }

      /**
       * Compute alpha at the given thickness
       */
      double compute_alpha(double thickness) const {
        return 
          alpha_a1_ * std::exp(-0.5 * std::pow((thickness - alpha_m1_) / alpha_s1_, 2)) +
          alpha_a2_ * std::exp(-0.5 * std::pow((thickness - alpha_m2_) / alpha_s2_, 2));
      }
      
      /**
       * Compute theta at the given thickness
       */
      double compute_theta(double thickness) const {
        return 
          theta_a1_ * std::exp(-0.5 * std::pow((thickness - theta_m1_) / theta_s1_, 2)) +
          theta_a2_ * std::exp(-0.5 * std::pow((thickness - theta_m2_) / theta_s2_, 2));
      }

      /**
       * Compute the mask
       */
      void compute_mask(double z_0, double z_e) {

        // Check the sizes match
        MULTEM_ASSERT(masker_.xsize() == grid_2d_.nx);
        MULTEM_ASSERT(masker_.ysize() == grid_2d_.ny);
        MULTEM_ASSERT(masker_.xsize() * masker_.ysize() == mask_.size());

        // Create host vector and then compute mask and copy
        mt::Vector<bool, mt::e_host> mask_host;
        mask_host.resize(mask_.size());
        masker_.compute(z_0, z_e, mask_host.begin());
        mask_.assign(mask_host.begin(), mask_host.end());
      }

      /**
       * Compute a gaussian random field
       */
      void compute_gaussian_random_field() {

        // We get two random fields for one calculation so we either use the
        // real or imaginary component 
        if (fft_data_use_real_) {
         
          // The data size
          std::size_t xsize = grid_2d_.nx;
          std::size_t ysize = grid_2d_.ny;
          std::size_t size = xsize*ysize;

          // Compute the Fourier transform of the Gaussian Random Field
          thrust::counting_iterator<size_t> indices(0);
          thrust::transform(
              indices,
              indices + size,
              fft_data_.begin(),
              ComputeGaussianRandomField<thrust::default_random_engine, T_c>(
                gen_, 
                a1_, 
                a2_, 
                a3_, 
                m1_, 
                m2_, 
                m3_, 
                s1_, 
                s2_, 
                s3_, 
                xsize, 
                ysize));
          gen_.discard(size);

          // Shift the FFT and then invert
          mt::fft2_shift(grid_2d_, fft_data_);
          fft_2d_->inverse(fft_data_);

          // Extract the real component
          mt::assign_real(fft_data_, random_field_);

        } else {
          
          // Extract the imag component
          mt::assign_imag(fft_data_, random_field_);
        }

        // Compute the mean
        double mean = thrust::reduce(
            random_field_.begin(),
            random_field_.end(), 
            double(0), 
            mt::functor::add<double>()) / random_field_.size();

        // Compute the variance
        double variance = thrust::transform_reduce(
            random_field_.begin(),
            random_field_.end(), 
            mt::functor::square_dif<double, double>(0), 
            double(0), 
            mt::functor::add<double>()) / random_field_.size();
        
        // Normalize by the variance
        thrust::transform(
            random_field_.begin(),
            random_field_.end(),
            random_field_.begin(),
            Normalize(mean, std::sqrt(variance)));
        /* mt::scale(1.0 / std::sqrt(variance), random_field_); */

        // Toggle real/imag
        fft_data_use_real_ = !fft_data_use_real_;
      }

      /**
       * Compute the Gamma random field and add to input potential
       */
      void compute_gamma_random_field(
          double alpha,
          double theta) {
        alpha = 1.279146202556906;
        theta = 9.458923282489229;
        //alpha = 1.059916553870336;
        //theta = 11.405600300446869;
        // Compute the lookup table for the quantile function of the Gamma distribution
        const size_t N_LOOKUP = 1000;
        mt::Vector<double, mt::e_host> gx(N_LOOKUP);
        mt::Vector<double, mt::e_host> gy(N_LOOKUP);
        mt::Vector<double, DeviceType> gx_dev(gx.size());
        mt::Vector<double, DeviceType> gy_dev(gy.size());
        for (std::size_t i = 0; i < gx.size(); ++i) {
          gx[i] = (double)i / (double)gx.size();
          gy[i] = boost::math::gamma_p_inv(alpha, gx[i]) * theta;
        }
        gx_dev.assign(gx.begin(), gx.end());
        gy_dev.assign(gy.begin(), gy.end());

        // Transform the Normal distribution into a Gamma distribution
        thrust::transform(
            random_field_.begin(), 
            random_field_.end(), 
            mask_.begin(),
            random_field_.begin(), 
            GaussianToGamma(
              thrust::raw_pointer_cast(gx_dev.data()),
              thrust::raw_pointer_cast(gy_dev.data()),
              gx.size())); 

        // Shift the grid
        mt::fft2_shift(grid_2d_, random_field_);
      }

      /**
       * Compute the Gamma random field and add to input potential
       */
      void operator()(
          double z_0,
          double z_e,
          mt::Vector<FloatType, DeviceType> &V_0) {
        
        std::cout << z_0 << ", " << z_e << ", " << (z_e-z_0) << std::endl;

        // Check the sizes
        MULTEM_ASSERT(z_0 < z_e);
        MULTEM_ASSERT(grid_2d_.nx > 0 && grid_2d_.ny > 0);
        MULTEM_ASSERT(fft_2d_ != NULL);
        MULTEM_ASSERT(V_0.size() == random_field_.size());
        MULTEM_ASSERT(mask_.size() == random_field_.size());

        // The gamma parameters given the slice slice thickness
        double thickness = (z_e - z_0);
        double alpha = compute_alpha(thickness);
        double theta = compute_theta(thickness);

        // Compute the mask
        compute_mask(z_0, z_e);

        // Compute the Fourier transform of the Gaussian Random Field
        compute_gaussian_random_field();

        // Convert the gaussian random field into a gamma field
        compute_gamma_random_field(alpha, theta);

        //mt::Vector<FloatType, mt::e_host> V(V_0.size());
        //V.assign(random_field_.begin(), random_field_.end());
        //static int index = 0;
        //std::string filename = "test_potential_" + std::to_string(index) + ".dat";
        //std::ifstream handle(filename);
        //for (auto &v : V) {
        //  handle >> v;
        //}
        //random_field_.assign(V.begin(), V.end());
        /* std::string filename = "statistical_potential_" + std::to_string(index) + ".dat"; */
        /* std::ofstream handle(filename); */
        /* handle << z_e - z_0 << std::endl; */
        /* for (auto v : V) { */
        /*   handle << v << ", "; */
        /* } */
        /* handle << std::endl; */
        //index++;
        /* exit(0); */
        // Add the random field to the potential map
        thrust::transform(
            random_field_.begin(), 
            random_field_.end(),
            V_0.begin(),
            V_0.begin(), 
            mt::functor::add<T_r>());
      }
    };
   

    /**
     * Run the multislice simulation
     * @param system_conf The system configuration
     * @param input_multislice The input object
     * @param output_multislice The output object
     */
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

      // If there was an error then throw an exception
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
        throw multem::Error(__FILE__, __LINE__, cudaGetErrorString(err));
      }
    }
    
    /**
     * Run the multislice simulation
     * @param system_conf The system configuration
     * @param input_multislice The input object
     * @param output_multislice The output object
     */
    template <typename FloatType, mt::eDevice DeviceType, typename Masker>
    void run_multislice_internal(
      const mt::System_Configuration &system_conf,
      mt::Input_Multislice<FloatType> &input_multislice,
      mt::Output_Multislice<FloatType> &output_multislice,
      const Masker &masker) {
  
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
     
      // Setup the ice potential approximation
      IcePotentialApproximation<FloatType, DeviceType> potential_function;
      potential_function.set_fft_2d(&fft_2d);
      potential_function.set_grid(input_multislice.grid_2d);
      potential_function.set_masker(masker);
      multislice.set_potential_function(&potential_function);

      // Set the input data
      output_multislice.set_input_data(&input_multislice);

      // Perform the multislice simulation
      multislice(output_multislice);
      stream.synchronize();

      // Get the results
      output_multislice.gather();
      output_multislice.clean_temporal();
      fft_2d.cleanup();

      // If there was an error then throw an exception
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
        throw multem::Error(__FILE__, __LINE__, cudaGetErrorString(err));
      }
    }
    
    /**
     * Run the multislice simulation
     * @param system_conf The system configuration
     * @param input_multislice The input object
     * @param output_multislice The output object
     */
    template <typename FloatType, mt::eDevice DeviceType>
    void run_projected_potential_internal(
      const mt::System_Configuration &system_conf,
      mt::Input_Multislice<FloatType> &input_multislice,
      mt::Output_Multislice<FloatType> &output_multislice,
      projected_potential_callback callback) {
  
      // Set the system configration    
      input_multislice.system_conf = system_conf;

      // Open a stream
      mt::Stream<DeviceType> stream(system_conf.nstream);

      // Setup the multislice simulation 
      mt::Projected_Potential<FloatType, DeviceType> projected_potential;
      projected_potential.set_input_data(&input_multislice, &stream);
      
      // Set the input data
      output_multislice.set_input_data(&input_multislice);

      // Perform the multislice simulation
      mt::Vector<FloatType, DeviceType> V(input_multislice.grid_2d.nxy());
      mt::Vector<FloatType, mt::e_host> V_host(input_multislice.grid_2d.nxy());
      for (auto islice = 0; islice < projected_potential.slicing.slice.size(); ++islice) {
        double z_0 = projected_potential.slicing.slice[islice].z_0;
        double z_e = projected_potential.slicing.slice[islice].z_e;
        projected_potential(islice, V);
        mt::fft2_shift(input_multislice.grid_2d, V);
        V_host.assign(V.begin(), V.end());
        callback(
            z_0, 
            z_e, 
            Image<double>(V_host.data(), 
              Image<double>::shape_type({
                input_multislice.grid_2d.nx,
                input_multislice.grid_2d.ny})));
      }

      // Syncronize stream
      stream.synchronize();
      
      // Get the results
      output_multislice.gather();
      output_multislice.clean_temporal();

      // If there was an error then throw an exception
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
        throw multem::Error(__FILE__, __LINE__, cudaGetErrorString(err));
      }
    }
 
    /**
     * Convert the multem::SystemConfiguration object to a
     * mt::System_Configuration object
     * @param config The multem::SystemConfiguration object
     * @returns The mt::System_Configuration object
     */
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
    
    /**
     * Convert the multem::Input object to a
     * mt::Input_Multislice object
     * @param config The multem::Input object
     * @returns The mt::Input_Multislice object
     */
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
          input_multislice.atoms.l_x = input.spec_lx;
          input_multislice.atoms.l_y = input.spec_ly;
          input_multislice.atoms.l_z = input.spec_lz;
          input_multislice.atoms.dz = input.spec_dz;
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
          MULTEM_ASSERT(input_multislice.thick.size() > 0);
        }

        // Potential slicing
        input_multislice.potential_slicing =
          detail::from_string<mt::ePotential_Slicing>(input.potential_slicing);
      }

      // XY sampling
      input_multislice.grid_2d.set_input_data(
          input.nx, 
          input.ny, 
          input.spec_lx, 
          input.spec_ly, 
          input.spec_dz, 
          input.bwl, 
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
      input_multislice.cond_lens.set_dsf_sigma(input.cond_lens_dsf_sigma);
      input_multislice.cond_lens.dsf_npoints = input.cond_lens_dsf_npoints;

      // source spread function
      input_multislice.cond_lens.set_ssf_sigma(input.cond_lens_ssf_sigma);
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
      input_multislice.obj_lens.set_dsf_sigma(input.obj_lens_dsf_sigma);
      input_multislice.obj_lens.dsf_npoints = input.obj_lens_dsf_npoints;

      // source spread function
      input_multislice.obj_lens.set_ssf_sigma(input_multislice.cond_lens.ssf_sigma);
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

    /**
     * Convert the mt::Output_Multislice object to a multem::Output object
     * @param output_multislice The mt::Output_Multislice object
     * @returns Output The multem::Output object
     */
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
                  Image<double>::shape_type({ nx, ny }));
            if (output_multislice.pn_coh_contrib) {
              result.data[i].image_coh[j] = Image<double>(
                  output_multislice.image_coh[i].image[j].data(), 
                    Image<double>::shape_type({ nx, ny }));
            }
          }
        }
      } else if (output_multislice.is_EWFS_EWRS()) {
        for (auto i = 0; i < output_multislice.thick.size(); ++i) {
          if (!output_multislice.is_EWFS_EWRS_SC()) {
            if (result.data.size() == output_multislice.m2psi_tot.size()) {
              result.data[i].m2psi_tot = Image<double>(
                  output_multislice.m2psi_tot[i].data(), 
                    Image<double>::shape_type({
                      (std::size_t) output_multislice.nx,
                      (std::size_t) output_multislice.ny}));
            }
          }
          if (result.data.size() == output_multislice.psi_coh.size()) {
            result.data[i].psi_coh = Image< std::complex<double> >(
                output_multislice.psi_coh[i].data(), 
                Image< std::complex<double> >::shape_type({
                  (std::size_t) output_multislice.nx,
                  (std::size_t) output_multislice.ny}));
          }
        }
      } else {
        for (auto i = 0; i < output_multislice.thick.size(); ++i) {
          if (result.data.size() == output_multislice.m2psi_tot.size()) {
            result.data[i].m2psi_tot = Image<double>(
                output_multislice.m2psi_tot[i].data(), 
                  Image<double>::shape_type({
                    (std::size_t) output_multislice.nx,
                    (std::size_t) output_multislice.ny}));
          }
          if (output_multislice.pn_coh_contrib) {
            if (result.data.size() == output_multislice.m2psi_coh.size()) {
              result.data[i].m2psi_coh = Image<double>(
                  output_multislice.m2psi_coh[i].data(), 
                  Image<double>::shape_type({
                    (std::size_t) output_multislice.nx,
                    (std::size_t) output_multislice.ny}));
            }
          }
        }
      }

      if (result.data.size() == output_multislice.V.size()) {
        for (auto i = 0; i < output_multislice.V.size(); ++i) {
          MULTEM_ASSERT(output_multislice.V[i].size() == output_multislice.nx*output_multislice.ny);
          result.data[i].V = Image<double>(
                output_multislice.V[i].data(), 
                Image<double>::shape_type({
                  (std::size_t) output_multislice.nx,
                  (std::size_t) output_multislice.ny}));
        }
      }


      // Return the result
      return result;
    }
  }

  /**
   * Run the multislice simulation. 
   * @param config The system configuration
   * @param input The input object
   * @returns The output results
   */
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
  
  /**
   * Run the multislice simulation. 
   * @param config The system configuration
   * @param input The input object
   * @returns The output results
   */
  template <typename FloatType, mt::eDevice DeviceType, typename Masker>
  Output run_multislice(SystemConfiguration config, Input input, const Masker &masker) {

    // Add a couple of hydrogen atoms at extreme z values. This is needed
    // because if multem has no atoms it gets confused. By adding an atom at
    // the minimum and maximum Z locations we make multem create slices between
    // those two atoms.
    input.spec_atoms.push_back(Atom(1, 0, 0, masker.zmin(), 0.085, 1, 0, 0));
    input.spec_atoms.push_back(Atom(1, 0, 0, masker.zmax(), 0.085, 1, 0, 0));

    // Initialise the system configuration and input structures 
    auto system_conf = detail::read_system_configuration(config);
    auto input_multislice = detail::read_input_multislice<FloatType>(input);
    input_multislice.system_conf = system_conf;

    // Create the output structure
    mt::Output_Multislice<FloatType> output_multislice;
   
    // Run the simulation 
    detail::run_multislice_internal<FloatType, DeviceType, Masker>(
      system_conf, input_multislice, output_multislice, masker);

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
      MULTEM_ASSERT(config.device == "host" || config.device == "device");
      MULTEM_ASSERT(config.precision == "float" || config.precision == "double");
    } 
    return result;
  }

  Output simulate_with_ice_approximation(
      SystemConfiguration config, 
      Input input, 
      const Masker &masker) {
    Output result;
    if (config.device == "host" && config.precision == "float") {
      result = run_multislice<float, mt::e_host, Masker>(config, input, masker);
    } else if (config.device == "host" && config.precision == "double") {
      result = run_multislice<double, mt::e_host, Masker>(config, input, masker);
    } else if (config.device == "device" && config.precision == "float") {
      result = run_multislice<float, mt::e_device, Masker>(config, input, masker);
    } else if (config.device == "device" && config.precision == "double") {
      result = run_multislice<double, mt::e_device, Masker>(config, input, masker);
    } else {
      MULTEM_ASSERT(config.device == "host" || config.device == "device");
      MULTEM_ASSERT(config.precision == "float" || config.precision == "double");
    } 
    return result;
  }
  
  /**
   * Run the multislice simulation. 
   * @param config The system configuration
   * @param input The input object
   * @returns The output results
   */
  template <typename FloatType, mt::eDevice DeviceType>
  Output run_projected_potential(
      SystemConfiguration config, 
      Input input,
      projected_potential_callback callback) {

    // Initialise the system configuration and input structures 
    auto system_conf = detail::read_system_configuration(config);
    auto input_multislice = detail::read_input_multislice<FloatType>(input);
    input_multislice.system_conf = system_conf;

    // Create the output structure
    mt::Output_Multislice<FloatType> output_multislice;
   
    // Run the simulation 
    detail::run_projected_potential_internal<FloatType, DeviceType>(
      system_conf, input_multislice, output_multislice, callback);

    // Return the output struct
    return detail::write_output_multislice(output_multislice);
  }

  Output compute_projected_potential(
      SystemConfiguration config, 
      Input input,
      projected_potential_callback callback) {
    Output result;
    if (config.device == "host" && config.precision == "float") {
      result = run_projected_potential<float, mt::e_host>(config, input, callback);
    } else if (config.device == "host" && config.precision == "double") {
      result = run_projected_potential<double, mt::e_host>(config, input, callback);
    } else if (config.device == "device" && config.precision == "float") {
      result = run_projected_potential<float, mt::e_device>(config, input, callback);
    } else if (config.device == "device" && config.precision == "double") {
      result = run_projected_potential<double, mt::e_device>(config, input, callback);
    } else {
      MULTEM_ASSERT(config.device == "host" || config.device == "device");
      MULTEM_ASSERT(config.precision == "float" || config.precision == "double");
    } 
    return result;
  }

  Image< std::complex<double> > compute_ctf(
      SystemConfiguration config, 
      Input input) {

    // Initialise the system configuration and input structures 
    config.device = "host";
    config.precision = "double";
    auto system_conf = detail::read_system_configuration(config);
    auto input_multislice = detail::read_input_multislice<double>(input);
    input_multislice.system_conf = system_conf;

    // Initialise the vectors
    mt::Vector< thrust::complex<double>, mt::e_host > psi_in(input_multislice.grid_2d.nxy());
    mt::Vector< thrust::complex<double>, mt::e_host > psi_out(input_multislice.grid_2d.nxy());
    psi_in.assign(psi_in.size(), 1.0);
      
    // Open a stream
    mt::Stream<mt::e_host> stream(system_conf.nstream);

    // Compute CTF
    switch (input_multislice.illumination_model) {
      case mt::eIM_Coherent:
				mt::apply_CTF(
            stream, 
            input_multislice.grid_2d, 
            input_multislice.obj_lens, 
            0, 
            0, 
            psi_in, 
            psi_out);
        break;
      case mt::eIM_Partial_Coherent:
        mt::apply_PCTF(
            stream, 
            input_multislice.grid_2d, 
            input_multislice.obj_lens, 
            psi_in, 
            psi_out);
        break;
      default:
        break;
    }

    // Syncronize stream
    stream.synchronize();

    // Return the image
    return Image< std::complex<double> >(
        psi_out.data(), 
        { 
          input_multislice.grid_2d.nx,
          input_multislice.grid_2d.ny 
        });
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


  /**
   * Test to ensure that ice potential approximation gives correct results
   */
  template <typename FloatType, mt::eDevice DeviceType>
  void test_ice_potential_approximation_internal() {

    // Create the grid
    mt::Grid_2d<FloatType> grid_2d(1000,1000);

    // Create the FFT
    mt::FFT<FloatType, DeviceType> fft_2d;
    fft_2d.create_plan_2d(grid_2d.ny, grid_2d.nx, 1);

    // Create the masker
    Masker masker(grid_2d.nx, grid_2d.ny, 1.0);
    masker.set_cube({0,0,0}, 1000);

    // Create the random generator
    thrust::default_random_engine gen;

    // Create the ice potential
    multem::detail::IcePotentialApproximation<FloatType, DeviceType> potential_function;
    potential_function.set_fft_2d(&fft_2d);
    potential_function.set_grid(grid_2d);
    potential_function.set_random_engine(gen);
    potential_function.set_masker(masker);

    // Compute
    mt::Vector<FloatType, DeviceType> V0_dev(grid_2d.nx*grid_2d.ny);
    potential_function(0, 5, V0_dev);
    mt::Vector<FloatType, mt::e_host> V0(V0_dev.begin(), V0_dev.end());

    // The CPU based function to see that we get the same result
    mt::Vector<FloatType, mt::e_host> V1;
    {

      // The size of the slice
      double z_0 = 0;
      double z_e = 5;

      // Compute the mask
      mt::Vector<bool, mt::e_host> mask;
      mask.resize(V0.size());
      masker.compute(z_0, z_e, mask.begin());
      
      // The parameters to use
      double alpha_a1 = 3.61556763;
      double alpha_a2 = 23.22955402;
      double alpha_m1 = 5.48214868;
      double alpha_m2 = 11.81498691;
      double alpha_s1 = 2.27209584;
      double alpha_s2 = 3.64439385;
      double theta_a1 = 41.7597107;
      double theta_a2 = 1077.04791;
      double theta_m1 = 0.604256237;
      double theta_m2 = -10.0000000;
      double theta_s1 = 1.65486734;
      double theta_s2 = 35.4955295;
      double a1 = 1560.27916;
      double a2 = 4.41780420;
      double a3 = 693.960558;
      double m1 = 0.254522845;
      double m2 = 10.7305321;
      double m3 = 0.308002600;
      double s1 = 0.213959063;
      double s2 = 0.231840410;
      double s3 = 0.662902509;

      // The slice thickness
      double t = z_e - z_0;
      double alpha = 
        alpha_a1 * std::exp(-0.5*(t-alpha_m1)*(t-alpha_m1)/(alpha_s1*alpha_s1)) + 
        alpha_a2 * std::exp(-0.5*(t-alpha_m2)*(t-alpha_m2)/(alpha_s2*alpha_s2)); 
      double theta = 
        theta_a1 * std::exp(-0.5*(t-theta_m1)*(t-theta_m1)/(theta_s1*theta_s1)) + 
        theta_a2 * std::exp(-0.5*(t-theta_m2)*(t-theta_m2)/(theta_s2*theta_s2)); 

      // Compute the Gamma random field
      mt::Vector<FloatType, DeviceType> random_field;
      random_field.resize(V0.size());
        
      // The size of the data
      std::size_t size = random_field.size();

      // Create a uniform distribution random number generator
      thrust::uniform_real_distribution<double> uniform(0, 2*M_PI);

      mt::Vector<complex<FloatType>, mt::e_host> fft_data;
      mt::Vector<complex<FloatType>, DeviceType> fft_data_dev;
      fft_data.resize(size);
      fft_data_dev.resize(size);
      std::size_t xsize = grid_2d.nx;
      std::size_t ysize = grid_2d.ny;
      for (std::size_t j = 0; j < ysize; ++j) {
        for (std::size_t i = 0; i < xsize; ++i) {
          double xd = (i-xsize/2.0)/(xsize/2.0);
          double yd = (j-ysize/2.0)/(ysize/2.0);
          double r = std::sqrt(xd*xd + yd*yd);
          double power = 
            a1 * std::exp(-0.5*(r - m1)*(r - m1) / (s1*s1)) + 
            a2 * std::exp(-0.5*(r - m2)*(r - m2) / (s2*s2)) + 
            a3 * std::exp(-0.5*(r - m3)*(r - m3) / (s3*s3)); 
          double amplitude = std::sqrt(power);
          double phase = uniform(gen);
          fft_data[i+j*xsize] = amplitude * std::exp(std::complex<double>(0, phase)); 
        }
      }
      fft_data_dev.assign(fft_data.begin(), fft_data.end());
      mt::fft2_shift(grid_2d, fft_data_dev);
      fft_data_dev[0] = 0;
      fft_2d.inverse(fft_data_dev);
      
      mt::assign_real(fft_data_dev, random_field);
      
      mt::Vector<FloatType, mt::e_host> random_field_host;
      random_field_host.resize(size);
      random_field_host.assign(random_field.begin(), random_field.end());

      double mean = 0;
      /* for(auto x : random_field_host) { */
      /*   mean += x; */
      /* } */
      /* mean /= random_field_host.size(); */
      double sdev = 0;
      for (auto x : random_field_host) {
        sdev += std::pow((x - mean), 2);
      }
      sdev = std::sqrt(sdev / random_field_host.size());
      for (auto &x : random_field_host) {
        x = (x - mean) / sdev;
      }

      std::vector<double> gx(1000);
      std::vector<double> gy(1000);
      for (std::size_t i = 0; i < gx.size(); ++i) {
        gx[i] = (double)i / (double)gx.size();
        gy[i] = boost::math::gamma_p_inv(alpha, gx[i]) * theta;
      }

      for (std::size_t j = 0; j < ysize; ++j) {
        for (std::size_t i = 0; i < xsize; ++i) {
          auto &x = random_field_host[i+j*xsize];
          if (mask[i+j*xsize]) {
            auto c = 0.5 * (1 + std::erf(x/std::sqrt(2)));

            double g = 0;
            int i = (std::size_t)std::floor(c * gx.size());
            if (i < 0) {
              i = 0;
            } else if (i >= gx.size()-1) {
              i = gx.size()-2;
            }
            auto gx0 = gx[i];
            auto gx1 = gx[i+1];
            auto gy0 = gy[i];
            auto gy1 = gy[i+1];
            g = gy0 + (gy1 - gy0) / (gx1 - gx0) * (c - gx0);

            x = g;
          } else {
            x = 0;
          }
        }
      }

      mt::fft2_shift(grid_2d, random_field_host);
      V1.assign(random_field_host.begin(), random_field_host.end());
    }

    for (std::size_t i = 0; i < V1.size(); ++i) {
      MULTEM_ASSERT(std::abs((V0[i] - V1[i])) < 1e-3);
    }

    // Cleanup
    fft_2d.cleanup();

    // If there was an error then throw an exception
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw multem::Error(__FILE__, __LINE__, cudaGetErrorString(err));
    }
  }

  /**
   * Run tests for all devices and precisions
   */
  void test_ice_potential_approximation() {
    test_ice_potential_approximation_internal<float, mt::e_host>();
    test_ice_potential_approximation_internal<double, mt::e_host>();
    test_ice_potential_approximation_internal<float, mt::e_device>();
    test_ice_potential_approximation_internal<double, mt::e_device>();
  }

	/**
   * Run tests for the cuboid masker
   */
	void test_cuboid_masker() {
    
    // Setup the masker
    Masker masker;
    masker.set_image_size(100, 200);
    masker.set_pixel_size(0.5);
    masker.set_cuboid({ 10, 11, 12 }, { 20, 30, 40 });
    masker.set_translation({4, 5, 6});
    masker.set_rotation({10, 11, 12}, 90);

    // Check stuff
    MULTEM_ASSERT(masker.xsize() == 100);
    MULTEM_ASSERT(masker.ysize() == 200);
    MULTEM_ASSERT(masker.image_size() == 200*100);
    MULTEM_ASSERT(masker.pixel_size() == 0.5);
    MULTEM_ASSERT(masker.shape() == Masker::Cuboid);
    MULTEM_ASSERT(masker.offset()[0] ==  10);
    MULTEM_ASSERT(masker.offset()[1] ==  11);
    MULTEM_ASSERT(masker.offset()[2] ==  12);
    MULTEM_ASSERT(masker.size()[0] == 20);
    MULTEM_ASSERT(masker.size()[1] == 30);
    MULTEM_ASSERT(masker.size()[2] == 40);
    MULTEM_ASSERT(masker.xmin() == (10 + 4));
    MULTEM_ASSERT(masker.ymin() == (11 - 40 + 5));
    MULTEM_ASSERT(masker.zmin() == (12 + 6));
    MULTEM_ASSERT(masker.xmax() == (10 + 20 + 4));
    MULTEM_ASSERT(masker.ymax() == (11 + 5));
    MULTEM_ASSERT(masker.zmax() == (12 + 30 + 6));
    MULTEM_ASSERT(masker.rotation_origin()[0] == 10);
    MULTEM_ASSERT(masker.rotation_origin()[1] == 11);
    MULTEM_ASSERT(masker.rotation_origin()[2] == 12);
    MULTEM_ASSERT(masker.rotation_angle() == 90);
    MULTEM_ASSERT(masker.translation()[0] == 4);
    MULTEM_ASSERT(masker.translation()[1] == 5);
    MULTEM_ASSERT(masker.translation()[2] == 6);

    // Compute the mask
    std::vector<bool> mask(masker.image_size());
    for (std::size_t z = 0; z < 100; ++z) {
      masker.compute(z, z+1, mask.begin());
      for (std::size_t j = 0; j < 200; ++j) {
        for (std::size_t i = 0; i < 100; ++i) {
          double x = i * masker.pixel_size();
          double y = j * masker.pixel_size();
          bool value = false;
          if (x >= masker.xmin() && x < masker.xmax() && 
              y >= masker.ymin() && y < masker.ymax() &&
              z >= masker.zmin() && z < masker.zmax()) {
            value = true;
          }
          MULTEM_ASSERT(mask[j+i*masker.ysize()] == value);
        }
      }
    }
	}

	/**
   * Run tests for the cylinder masker
   */
  void test_cylinder_masker() {

    // Setup the masker
    Masker masker;
    masker.set_image_size(100, 200);
    masker.set_pixel_size(0.5);
    masker.set_cylinder({ 10, 11, 12 }, 50, 12);
    masker.set_translation({4, 5, 6});
    masker.set_rotation({10, 11, 12}, 90);

    // Check stuff
    MULTEM_ASSERT(masker.xsize() == 100);
    MULTEM_ASSERT(masker.ysize() == 200);
    MULTEM_ASSERT(masker.image_size() == 200*100);
    MULTEM_ASSERT(masker.pixel_size() == 0.5);
    MULTEM_ASSERT(masker.shape() == Masker::Cylinder);
    MULTEM_ASSERT(masker.offset()[0] ==  10);
    MULTEM_ASSERT(masker.offset()[1] ==  11);
    MULTEM_ASSERT(masker.offset()[2] ==  12);
    MULTEM_ASSERT(masker.size()[0] == 50);
    MULTEM_ASSERT(masker.size()[1] == 2*12);
    MULTEM_ASSERT(masker.size()[2] == 2*12);
    MULTEM_ASSERT(masker.xmin() == (10 + 4));
    MULTEM_ASSERT(masker.ymin() == (11 - 2*12 + 5));
    MULTEM_ASSERT(masker.zmin() == (12 + 6));
    MULTEM_ASSERT(masker.xmax() == (10 + 50 + 4));
    MULTEM_ASSERT(masker.ymax() == (11 + 5));
    MULTEM_ASSERT(masker.zmax() == (12 + 2*12 + 6));
    MULTEM_ASSERT(masker.rotation_origin()[0] == 10);
    MULTEM_ASSERT(masker.rotation_origin()[1] == 11);
    MULTEM_ASSERT(masker.rotation_origin()[2] == 12);
    MULTEM_ASSERT(masker.rotation_angle() == 90);
    MULTEM_ASSERT(masker.translation()[0] == 4);
    MULTEM_ASSERT(masker.translation()[1] == 5);
    MULTEM_ASSERT(masker.translation()[2] == 6);

    // Compute the mask
    std::vector<bool> mask(masker.image_size());
    for (std::size_t z = 0; z < 100; ++z) {
      masker.compute(z, z+1, mask.begin());
      for (std::size_t j = 0; j < 200; ++j) {
        for (std::size_t i = 0; i < 100; ++i) {
          double x = i * masker.pixel_size();
          double y = (j + 0.5) * masker.pixel_size();
          double yc = (masker.ymin() + masker.ymax()) / 2.0;
          double zc = (masker.zmin() + masker.zmax()) / 2.0;
          double d1 = std::sqrt((y-yc)*(y-yc)+(z-zc)*(z-zc));
          double d2 = std::sqrt((y-yc)*(y-yc)+(z+1-zc)*(z+1-zc));
          double radius = 12;
          bool value = false;
          if (x >= masker.xmin() && x < masker.xmax() && 
              std::min(d1,d2) < radius) {
            value = true;
          }
          MULTEM_ASSERT(mask[j+i*masker.ysize()] == value);
        }
      }
    }

  }

	/**
   * Run tests for the masker
   */
	void test_masker() {
		test_cuboid_masker();
		test_cylinder_masker();
	}
}

