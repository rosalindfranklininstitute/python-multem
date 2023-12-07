/*
 *  masker.h
 *
 *  Copyright (C) 2019 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the GPLv3 license, a copy of 
 *  which is included in the root directory of this package.
 */

#ifndef MULTEM_MASKER_H
#define MULTEM_MASKER_H

#include <array>
#include <cassert>
#include <complex>
#include <string>
#include <vector>
#include <multem/error.h>

namespace multem {

  namespace detail {
   
    /**
     * Subtract origin rotate and then add origin
     */
    inline
    void rotate2d_around_origin(
        double ox, 
        double oy, 
        double theta, 
        double &x, 
        double &y) {
      double xt = std::cos(theta) * (x-ox) - std::sin(theta) * (y-oy) + ox;
      double yt = std::sin(theta) * (x-ox) + std::cos(theta) * (y-oy) + oy;
      x = xt;
      y = yt;
    }

    /**
     * Compute the dot product between two vectors
     */
    template <typename Vector>
    double dot(Vector a, Vector b) {
      return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }

    /**
     * Compute the norm squared of a vector
     */
    template <typename Vector>
    double norm_sq(Vector a) {
      return dot(a, a);
    }

    /**
     * Compute the norm of a vector
     */
    template <typename Vector>
    double norm(Vector a) {
      return std::sqrt(detail::norm_sq(a));
    }

    /**
     * Compute the cross product of a vector
     */
    template <typename Vector>
    Vector cross(Vector a, Vector b) {
      return Vector({
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
      });
    }

    /**
     * Get the rotation between two vectors
     */
    template <typename Vector>
    Vector get_rotation_between_vectors(
        Vector a,
        Vector b) {
      
      // Compute length of vectors and normalize
      double la = detail::norm(a);
      double lb = detail::norm(b);
      MULTEM_ASSERT(la > 0 && lb > 0);
      a[0] /= la; a[1] /= la; a[2] /= la;
      b[0] /= lb; b[1] /= lb; b[2] /= lb;

      // Take the cross product to compute the rotation vector
      Vector r = detail::cross(b, a);

      // Compute angle between the vectors
      double theta = std::acos(detail::dot(b, a));
      if (theta < 0) {
        theta += 2*M_PI;
      }
      
      // Compute length of the rotation vector and return
      double lr = detail::norm(r);
      if (lr > 0) {
        r[0] = r[0] * theta / lr; 
        r[1] = r[1] * theta / lr; 
        r[2] = r[2] * theta / lr;
      }
      return r;
    }

    /**
     * Use rodriguez's formula to rotate a vector
     */
    template <typename Vector>
    Vector rotate(Vector k, Vector v) {
      double theta = detail::norm(k);
      if (theta > 0) {
        k[0] /= theta;
        k[1] /= theta;
        k[2] /= theta;
        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);
        double kdv = detail::dot(k, v);
        Vector kxv = detail::cross(k, v);
        v[0] = v[0] * cos_theta + kxv[0] * sin_theta + k[0] * kdv * (1 - cos_theta);
        v[1] = v[1] * cos_theta + kxv[1] * sin_theta + k[1] * kdv * (1 - cos_theta);
        v[2] = v[2] * cos_theta + kxv[2] * sin_theta + k[2] * kdv * (1 - cos_theta);
      }
      return v;
    }


    /**
     * Perform Hermite spline interpolation
     */
    class Spline {
    public:

      /**
       * Initialise the spline assuming uniform separation of points
       */
      template <typename Iterator>
      Spline(Iterator first, Iterator last)
      {
        std::size_t n = std::distance(first, last);
        MULTEM_ASSERT(n > 0);
        for (auto i = 0; i < n; ++i, ++first) {
          double A = *first;
          double B = *first;
          double C = *first;
          double D = *first;
          if (i > 0) {
            A = *(first - 1);
          }
          if (i < n - 1) {
            C = *(first + 1);
            D = *(first + 1);
          }
          if (i < n - 2) {
            D = *(first + 2);
          }
          compute_coefficients(A, B, C, D);
          a_.push_back(A);
          b_.push_back(B);
          c_.push_back(C);
          d_.push_back(D);
        }
      }

      /**
       * Interpolate the spline
       *
       * If t == 0 then the value will be Y(first) if t == 1.0 the value will
       * be Y(last)
       */
      template <typename T>
      double interpolate(T u) const {
        u = u * (size() - 1);
        int i = (int)std::floor(u);
        i = std::max(0, std::min((int)size()-1, i));
        MULTEM_ASSERT(i >= 0 && i < size())
        double t = u - i;
        return a_[i]*t*t*t + b_[i]*t*t + c_[i]*t + d_[i];
      }

      /**
       * The number of points in the spline 
       */
      std::size_t size() const {
        return a_.size();
      }

      /**
       * Get the parameters
       */
      std::vector<double> a() const {
        return a_;
      }

      /**
       * Get the parameters
       */
      std::vector<double> b() const {
        return b_;
      }
      
      /**
       * Get the parameters
       */
      std::vector<double> c() const {
        return c_;
      }
      
      /**
       * Get the parameters
       */
      std::vector<double> d() const {
        return d_;
      }

    protected:

      /**
       * Compute the spline coefficients
       */
      void compute_coefficients(double &A, double &B, double &C, double &D) const {
        double a = -A/2.0 + (3.0*B)/2.0 - (3.0*C)/2.0 + D/2.0;
        double b = A - (5.0*B)/2.0 + 2.0*C - D / 2.0;
        double c = -A/2.0 + C/2.0;
        double d = B;
        A = a;
        B = b;
        C = c;
        D = d;
      }

      std::vector<double> a_;
      std::vector<double> b_;
      std::vector<double> c_;
      std::vector<double> d_;
    };



  }

  /**
   * A cuboid masker
   */
  class CuboidMasker {
  public:

    using vector2 = std::array<double, 2>;
    using vector3 = std::array<double, 3>;
    
    CuboidMasker()
      : xsize_(0),
        ysize_(0),
        pixel_size_(1),
        offset_({0, 0, 0}),
        size_({0, 0, 0}),
        translation_({ 0, 0, 0 }),
        rotation_centre_({0, 0, 0}),
        rotation_vector_({0, 0, 0}) {
      update_geometry();    
    }

    /**
     * Get the x size of the mask image
     */
    std::size_t xsize() const {
      return xsize_;
    }

    /**
     * Get the y size of the mask image
     */
    std::size_t ysize() const {
      return ysize_;
    }

    /**
     * The total size
     */
    std::size_t image_size() const {
      return xsize() * ysize();
    }

    /**
     * Get the pixel size
     */
    double pixel_size() const {
      return pixel_size_;
    }

    /**
     * Get the offset
     */
    const vector3& offset() const {
      return offset_;
    }

    /**
     * Get the size
     */
    const vector3& size() const {
      return size_;
    }

    /**
     * Get the minimum x coordinate
     */
    double xmin() const {
      return pmin_[0];
    }
    
    /**
     * Get the minimum y coordinate
     */
    double ymin() const {
      return pmin_[1];
    }

    /**
     * Get the minimum z coordinate
     */
    double zmin() const {
      return pmin_[2];
    }
    
    /**
     * Get the maximum x coordinate
     */
    double xmax() const {
      return pmax_[0];
    }
    
    /**
     * Get the maximum y coordinate
     */
    double ymax() const {
      return pmax_[1];
    }

    /**
     * Get the maximum z coordinate
     */
    double zmax() const {
      return pmax_[2];
    }
    
    /**
     * Get the rotation origin
     */
    const vector3& rotation_origin() const {
      return rotation_centre_;
    }
    
    /**
     * Get the rotation angle
     */
    double rotation_angle() const {
      return detail::norm(rotation_vector_);
    }
    
    /**
     * Get the rotation vector
     */
    const vector3& rotation_vector() const {
      return rotation_vector_;
    }

    /**
     * Get the translation
     */
    const vector3& translation() const {
      return translation_;
    }
    
    /**
     * Get the points
     */
    const std::array<vector3, 8>& points() const {
      return points_;
    }

    /**
     * Set the image size
     */
    void set_image_size(std::size_t xsize, std::size_t ysize) {
      MULTEM_ASSERT(xsize > 0);
      MULTEM_ASSERT(ysize > 0);
      xsize_ = xsize;
      ysize_ = ysize;
      update_geometry();
    }
    
    /**
     * Set the pixel size
     */
    void set_pixel_size(double pixel_size) {
      MULTEM_ASSERT(pixel_size > 0);
      pixel_size_ = pixel_size;
      update_geometry();
    }
    
    /**
     * Set a cuboid mask
     */
    void set_parameters(std::array<double,3> offset, std::array<double,3> length) {
      MULTEM_ASSERT(length[0] > 0);
      MULTEM_ASSERT(length[1] > 0);
      MULTEM_ASSERT(length[2] > 0);
      offset_ = offset;
      size_ = length;
      update_geometry();
    }

    /**
     * Set a translation along the x axis
     */
    void set_translation(std::array<double, 3> translation) {
      translation_ = translation;
      update_geometry();
    }
    
    /**
     * Set a rotation about the x axis
     */
    void set_rotation(std::array<double, 3> centre, std::array<double, 3> rotvec) {
      rotation_centre_ = centre;
      rotation_vector_ = rotvec;
      update_geometry();
    }

    /**
     * Update the geometry
     */
    void update_geometry() {
      
      // The sample coordinate system can be rotated relative to the
      // microscope coordinate system by a rotation defined by the input
      // rotation vector and the centre of rotation. A point in the microscope
      // coordinate system can be found by rotating the sample about the
      // rotation centre
      //
      // point(microscope) = Rs * (point(sample) - centre) + centre + translation
      vector3 Rs = rotation_vector_;
      
      // The extreme points
      double x0 = offset_[0];
      double y0 = offset_[1];
      double z0 = offset_[2];
      double x1 = x0 + size_[0];
      double y1 = y0 + size_[1];
      double z1 = z0 + size_[2];
     
      // The x/y/z coords of the cuboid
      points_[0] = { x0, y0, z0 };
      points_[1] = { x1, y0, z0 };
      points_[2] = { x0, y1, z0 };
      points_[3] = { x0, y0, z1 };
      points_[4] = { x0, y1, z1 };
      points_[5] = { x1, y0, z1 };
      points_[6] = { x1, y1, z0 };
      points_[7] = { x1, y1, z1 };

      // Transform the points at the corners of the cuboid
      for (auto i = 0; i < points_.size(); ++i) {
        points_[i] = transform_cuboid_to_microscope(points_[i], Rs); 
      }

      // Compute the max and min
      pmin_ = { 1e10, 1e10, 1e10 };
      pmax_ = { -1e10, -1e10, -1e10 };
      for (auto i = 0; i < points_.size(); ++i) {
        pmin_[0] = std::min(pmin_[0], points_[i][0]);
        pmin_[1] = std::min(pmin_[1], points_[i][1]);
        pmin_[2] = std::min(pmin_[2], points_[i][2]);
        pmax_[0] = std::max(pmax_[0], points_[i][0]);
        pmax_[1] = std::max(pmax_[1], points_[i][1]);
        pmax_[2] = std::max(pmax_[2], points_[i][2]);
      }
    }
    
    /**
     * Compute the cuboid mask
     */
    template <typename Iterator>
    void compute_mask(double zs, double ze, Iterator iterator) const {
      MULTEM_ASSERT(ze > zs);

      // The middle z coordinate of the slice
      double zc = (zs + ze) / 2.0;
        
      // Get the points defining the cuboid
      auto p1 = points_[0];
      auto p2 = points_[1];
      auto p3 = points_[2];
      auto p4 = points_[3];

      // Only do something if the slice is within range
      if (zs < zmax() && ze > zmin()) {

        // Vector P1 -> P2
        auto u0 = p2[0] - p1[0];
        auto u1 = p2[1] - p1[1];
        auto u2 = p2[2] - p1[2];
        
        // Vector P1 -> P3
        auto v0 = p3[0] - p1[0];
        auto v1 = p3[1] - p1[1];
        auto v2 = p3[2] - p1[2];
        
        // Vector P1 -> P4
        auto w0 = p4[0] - p1[0];
        auto w1 = p4[1] - p1[1];
        auto w2 = p4[2] - p1[2];
       
        // Dot products u.p1, u.p2, v.p1, v.p3, w.p1, w.p4
        auto u_p1 = u0*p1[0] + u1*p1[1] + u2*p1[2];
        auto u_p2 = u0*p2[0] + u1*p2[1] + u2*p2[2];
        auto v_p1 = v0*p1[0] + v1*p1[1] + v2*p1[2];
        auto v_p3 = v0*p3[0] + v1*p3[1] + v2*p3[2];
        auto w_p1 = w0*p1[0] + w1*p1[1] + w2*p1[2];
        auto w_p4 = w0*p4[0] + w1*p4[1] + w2*p4[2];

        // order the tests
        if (u_p1 > u_p2) std::swap(u_p1, u_p2);
        if (v_p1 > v_p3) std::swap(v_p1, v_p3);
        if (w_p1 > w_p4) std::swap(w_p1, w_p4);

        // Compute the slice mask
        for (std::size_t i = 0; i < xsize_; ++i) {
          for (std::size_t j = 0; j < ysize_; ++j) {

            // The coordinate in microscope scape
            double x = (i + 0.5) * pixel_size_;
            double y = (j + 0.5) * pixel_size_;
            double z = zc;


            // Dot product u.x, v.x and w.x
            double u_x = u0*x + u1*y + u2*z;
            double v_x = v0*x + v1*y + v2*z;
            double w_x = w0*x + w1*y + w2*z;
      
            // Dot product must be between bounds
            if (((u_x >= u_p1) && (u_x < u_p2)) &&
                ((v_x >= v_p1) && (v_x < v_p3)) &&
                ((w_x >= w_p1) && (w_x < w_p4))) {
              *iterator = true;
            } else {
              *iterator = false;
            }
            ++iterator;
          }
        }
      } else {
        std::fill(iterator, iterator + image_size(), 0);
      }
    }

  protected:
    
    /**
     * Transform a coordinate from cuboid coordinate system to microscope
     * coordinate system
     */
    vector3 transform_cuboid_to_microscope(vector3 a, vector3 Rs) const {
        auto b = a;
        b[0] -= rotation_centre_[0];
        b[1] -= rotation_centre_[1];
        b[2] -= rotation_centre_[2];
        b = detail::rotate(Rs, b);
        b[0] += rotation_centre_[0] + translation_[0];
        b[1] += rotation_centre_[1] + translation_[1];
        b[2] += rotation_centre_[2] + translation_[2];
        return b;
    }

    std::size_t xsize_;
    std::size_t ysize_;
    double pixel_size_;
    vector3 offset_;
    vector3 size_;
    vector3 translation_;
    vector3 rotation_centre_;
    vector3 rotation_vector_;
    std::array<vector3, 8> points_;
    vector3 pmin_;
    vector3 pmax_;
  };


  /**
   * A cylinder masker
   */
  class CylinderMasker {
  public:

    class Parameters {
    public:
      Parameters()
        : x_a(0),
          x_b(0),
          x_c(0),
          x_d(0),
          y_a(0),
          y_b(0),
          y_c(0),
          y_d(0),
          z_a(0),
          z_b(0),
          z_c(0),
          z_d(0),
          r_a(0),
          r_b(0),
          r_c(0),
          r_d(0) {}

      Parameters(
          double x_a_,
          double x_b_,
          double x_c_,
          double x_d_,
          double y_a_,
          double y_b_,
          double y_c_,
          double y_d_,
          double z_a_,
          double z_b_,
          double z_c_,
          double z_d_,
          double r_a_,
          double r_b_,
          double r_c_,
          double r_d_)
        : x_a(x_a_),
          x_b(x_b_),
          x_c(x_c_),
          x_d(x_d_),
          y_a(y_a_),
          y_b(y_b_),
          y_c(y_c_),
          y_d(y_d_),
          z_a(z_a_),
          z_b(z_b_),
          z_c(z_c_),
          z_d(z_d_),
          r_a(r_a_),
          r_b(r_b_),
          r_c(r_c_),
          r_d(r_d_) {}

      double x_a;
      double x_b;
      double x_c;
      double x_d;
      double y_a;
      double y_b;
      double y_c;
      double y_d;
      double z_a;
      double z_b;
      double z_c;
      double z_d;
      double r_a;
      double r_b;
      double r_c;
      double r_d;
    };

    using vector3 = std::array<double, 3>;
    
    CylinderMasker()
      : xsize_(0),
        ysize_(0),
        pixel_size_(1),
        length_(0),
        origin_({0, 0, 0}),
        axis_({0, 1, 0}),
        x_offset_({0}),
        z_offset_({0}),
        radius_({1}),
        translation_({ 0, 0, 0 }),
        rotation_centre_({0, 0, 0}),
        rotation_vector_({0, 0, 0}),
        A_({0, 0, 0}),
        B_({0, 0, 0}),
        zmin_(0),
        zmax_(0) {
      update_geometry();    
    }

    /**
     * Get the x size of the mask image
     */
    std::size_t xsize() const {
      return xsize_;
    }

    /**
     * Get the y size of the mask image
     */
    std::size_t ysize() const {
      return ysize_;
    }

    /**
     * The total size
     */
    std::size_t image_size() const {
      return xsize() * ysize();
    }

    /**
     * Get the pixel size
     */
    double pixel_size() const {
      return pixel_size_;
    }

    /**
     * The start of the cylinder
     */
    vector3 A() const {
      return A_;
    }
    
    /**
     * The end of the cylinder
     */
    vector3 B() const {
      return B_;
    }

    /**
     * Get the minimum x coordinate
     */
    double xmin() const {
      return A_[0];
    }
    
    /**
     * Get the minimum y coordinate
     */
    double ymin() const {
      return A_[1];
    }

    /**
     * Get the minimum z coordinate
     */
    double zmin() const {
      return zmin_;
    }
    
    /**
     * Get the maximum x coordinate
     */
    double xmax() const {
      return B_[0];
    }
    
    /**
     * Get the maximum y coordinate
     */
    double ymax() const {
      return B_[1];
    }

    /**
     * Get the maximum z coordinate
     */
    double zmax() const {
      return zmax_;
    }
    
    /**
     * Get the rotation origin
     */
    const vector3& rotation_origin() const {
      return rotation_centre_;
    }
    
    /**
     * Get the rotation vector
     */
    const vector3& rotation_vector() const {
      return rotation_vector_;
    }
    
    /**
     * Get the rotation angle
     */
    double rotation_angle() const {
      return detail::norm(rotation_vector_);
    }

    /**
     * Get the translation
     */
    const vector3& translation() const {
      return translation_;
    }

    /**
     * Get the cylinder parameters
     */
    const std::array<CylinderMasker::Parameters, 10>& parameters() const {
      return params_;
    }

    /**
     * Set the image size
     */
    void set_image_size(std::size_t xsize, std::size_t ysize) {
      MULTEM_ASSERT(xsize > 0);
      MULTEM_ASSERT(ysize > 0);
      xsize_ = xsize;
      ysize_ = ysize;
      update_geometry();
    }
    
    /**
     * Set the pixel size
     */
    void set_pixel_size(double pixel_size) {
      MULTEM_ASSERT(pixel_size > 0);
      pixel_size_ = pixel_size;
      update_geometry();
    }
    
    /**
     * Set a cylinder parameters
     */
    void set_parameters(
        std::array<double, 3> origin, 
        std::array<double, 3> axis,
        double length,
        std::vector<double> radius,
        std::vector<double> x_offset,
        std::vector<double> z_offset) {
      MULTEM_ASSERT(length > 0);
      MULTEM_ASSERT(radius.size() == 1 || radius.size() <= 10);
      MULTEM_ASSERT(radius.size() == x_offset.size());
      MULTEM_ASSERT(radius.size() == z_offset.size());
      origin_ = origin;
      axis_ = axis;
      length_ = length;
      if (radius.size() == 1) {
        radius_.fill(radius[0]);
        x_offset_.fill(x_offset[0]);
        z_offset_.fill(z_offset[0]);
      } else if (radius.size() == 10) {
        std::copy(radius.begin(), radius.end(), radius_.begin());
        std::copy(x_offset.begin(), x_offset.end(), x_offset_.begin());
        std::copy(z_offset.begin(), z_offset.end(), z_offset_.begin());
      } else {
        auto a = (double)(radius.size() - 1) / (double)(radius_.size() - 1);
        for (auto i = 0 ; i < radius_.size(); ++i) {
          auto j = a * i;
          if (j <= 0) {
            radius_[i] = radius[0];
            x_offset_[i] = x_offset[0];
            z_offset_[i] = z_offset[0];
          } else if (j >= radius.size() - 1) {
            radius_[i] = radius[radius.size() - 1];
            x_offset_[i] = x_offset[radius.size() - 1];
            z_offset_[i] = z_offset[radius.size() - 1];
          } else {
            auto j0 = (std::size_t)std::floor(j);
            auto j1 = (std::size_t)std::ceil(j);
            auto t = j1 - j0;
            radius_[i] = (1 - t) * radius[j0] + t * radius[j1];
            x_offset_[i] = (1 - t) * x_offset[j0] + t * x_offset[j1];
            z_offset_[i] = (1 - t) * z_offset[j0] + t * z_offset[j1];
          }
        }
      }
      update_geometry();
    }

    /**
     * Set a translation along the x axis
     */
    void set_translation(std::array<double, 3> translation) {
      translation_ = translation;
      update_geometry();
    }
    
    /**
     * Set a rotation about an axis
     */
    void set_rotation(std::array<double, 3>centre, std::array<double, 3> rotvec) {
      rotation_centre_ = centre;
      rotation_vector_ = rotvec;
      update_geometry();
    }

    /**
     * Update the geometry
     */
    void update_geometry() {
      
      // The cylinder coordinate system can be both offset and rotated
      // relative to the sample coordinate system.  The offset is given by the
      // origin variable and the rotation defined by the matrix needed to
      // rotate the y axis of the sample coordinate system to the cylinder
      // coordinate system. Compute the rotation matrix needed to rotate the
      // Y-axis of the sample coordinates into Cylinder coordinates defined by
      // the axis direction of the cylinder.
      //
      // point(sample) = Rc * point(cylinder) + origin
      vector3 Rc = detail::get_rotation_between_vectors(axis_, vector3({0, 1, 0}));

      // The sample coordinate system can be rotated relative to the
      // microscope coordinate system by a rotation defined by the input
      // rotation vector and the centre of rotation. A point in the microscope
      // coordinate system can be found by rotating the sample about the
      // rotation centre
      //
      // point(microscope) = Rs * (point(sample) - centre) + centre + translation
      vector3 Rs = rotation_vector_;
      
      // Transform the points at the beginning and end of the cylinder and
      // rotate the X and Z axes of the cylinder coordinate system
      A_ = transform_cylinder_to_microscope(vector3({0, 0, 0}), Rc, Rs);
      B_ = transform_cylinder_to_microscope(vector3({0, length_, 0}), Rc, Rs);
      auto X = detail::rotate(Rs, detail::rotate(Rc, vector3({1, 0, 0})));
      auto Z = detail::rotate(Rs, detail::rotate(Rc, vector3({0, 0, 1})));

      // Now compute the cylinder offsets in microscope coordinates
      MULTEM_ASSERT(x_offset_.size() == z_offset_.size());
      MULTEM_ASSERT(x_offset_.size() == radius_.size());
      std::vector<double> Ox(x_offset_.size());
      std::vector<double> Oy(x_offset_.size());
      std::vector<double> Oz(x_offset_.size());
      for (auto i = 0; i < Ox.size(); ++i) {
        Ox[i] = x_offset_[i] * X[0] + z_offset_[i] * Z[0];
        Oy[i] = x_offset_[i] * X[1] + z_offset_[i] * Z[1];
        Oz[i] = x_offset_[i] * X[2] + z_offset_[i] * Z[2];
      }

      // Interpolate the cylinder offsets and radii between the given points
      detail::Spline interpolate_x(Ox.begin(), Ox.end());
      detail::Spline interpolate_y(Oy.begin(), Oy.end());
      detail::Spline interpolate_z(Oz.begin(), Oz.end());
      detail::Spline interpolate_r(radius_.begin(), radius_.end());

      // Compute Z min and max
      zmin_ = 1e10;
      zmax_ = -1e10;
      std::size_t num_points = 1000;
      for (auto i = 0; i < num_points; ++i) {
        double t = (double)i / (num_points-1);
        double Ox = interpolate_x.interpolate(t);
        double Oy = interpolate_y.interpolate(t);
        double Oz = interpolate_z.interpolate(t);
        double r = interpolate_r.interpolate(t);
        double x = A_[0] + t * (B_[0] - A_[0]) + Oz;
        double y = A_[1] + t * (B_[1] - A_[1]) + Oz;
        double z = A_[2] + t * (B_[2] - A_[2]) + Oz;
        zmin_ = std::min(zmin_, z - r);
        zmax_ = std::max(zmax_, z + r);
      }
      MULTEM_ASSERT(zmin_ <= zmax_);

      // Add the interpolation parameters
      MULTEM_ASSERT(interpolate_x.size() == 10);
      for (std::size_t i = 0; i < interpolate_x.size(); ++i) {
        params_[i] = CylinderMasker::Parameters(
            interpolate_x.a()[i],
            interpolate_x.b()[i],
            interpolate_x.c()[i],
            interpolate_x.d()[i],
            interpolate_y.a()[i],
            interpolate_y.b()[i],
            interpolate_y.c()[i],
            interpolate_y.d()[i],
            interpolate_z.a()[i],
            interpolate_z.b()[i],
            interpolate_z.c()[i],
            interpolate_z.d()[i],
            interpolate_r.a()[i],
            interpolate_r.b()[i],
            interpolate_r.c()[i],
            interpolate_r.d()[i]);
      }
    }
    
    /**
     * Compute the cylinder mask
     */
    template <typename Iterator>
    void compute_mask(double zs, double ze, Iterator iterator) const {
      MULTEM_ASSERT(ze > zs);

      // Extract the coordinates of the end points of the cylinder
      double Ax = A_[0];
      double Ay = A_[1];
      double Az = A_[2];
      double Bx = B_[0];
      double By = B_[1];
      double Bz = B_[2];

      for (std::size_t i = 0; i < xsize_; ++i) {
        for (std::size_t j = 0; j < ysize_; ++j) {

          // The coordinate in microscope space
          double Px = (i + 0.5) * pixel_size_;
          double Py = (j + 0.5) * pixel_size_;
          double Pz = (zs + ze) / 2.0;

          // Compute the position along the cylinder
          //
          // t = (P-A).(B-A) / |B-A|^2
          double t = ((Px - Ax) * (Bx - Ax) + 
                      (Py - Ay) * (By - Ay) + 
                      (Pz - Az) * (Bz - Az)) / (length_*length_); 

          // Compute the offset and radius which is a function of the distance
          // along the cylinder. Get the index and position to interpolate
          double u = t * (params_.size() - 1);
          int index = std::max(0, std::min((int)params_.size()-1, (int)std::floor(u)));
          MULTEM_ASSERT(index >= 0 && index < params_.size())
          u = u - index;
          
          // Compute the offset and radius
          const CylinderMasker::Parameters& p = params_[index];
          double u2 = u*u;
          double u3 = u2*u;
          double x_offset = p.x_a*u3 + p.x_b*u2 + p.x_c*u + p.x_d;
          double y_offset = p.y_a*u3 + p.y_b*u2 + p.y_c*u + p.y_d;
          double z_offset = p.z_a*u3 + p.z_b*u2 + p.z_c*u + p.z_d;
          double radius   = p.r_a*u3 + p.r_b*u2 + p.r_c*u + p.r_d;

          // Compute the point along the cylinder and then the distance to the
          // cylinder. The offset here is because we can incorporate local
          // deformations of the cylinder as a function of the distance along
          // the axis of the cylinder
          //
          // C = A + t * (B - A)
          // d =  | P - C |
          double Cx = Ax + t * (Bx - Ax) + x_offset;
          double Cy = Ay + t * (By - Ay) + y_offset;
          double Cz = Az + t * (Bz - Az) + z_offset;
          double d2 = (Px - Cx)*(Px - Cx) + (Py - Cy)*(Py - Cy) + (Pz - Cz)*(Pz - Cz);

          // Set if the point is within the cylinder
          *iterator++ = (t >= 0) && (t < 1) && (d2 <= (radius*radius));
        }
      }
    }

  protected:

    /**
     * Transform a coordinate from cylinder coordinate system to microscope
     * coordinate system
     */
    vector3 transform_cylinder_to_microscope(vector3 a, vector3 Rc, vector3 Rs) const {
        vector3 b = detail::rotate(Rc, a);
        b[0] += origin_[0] - rotation_centre_[0];
        b[1] += origin_[1] - rotation_centre_[1];
        b[2] += origin_[2] - rotation_centre_[2];
        b = detail::rotate(Rs, b);
        b[0] += rotation_centre_[0] + translation_[0];
        b[1] += rotation_centre_[1] + translation_[1];
        b[2] += rotation_centre_[2] + translation_[2];
        return b;
    }

    std::size_t xsize_;             // The x size of the image
    std::size_t ysize_;             // The y size of the image
    double pixel_size_;             // The pixel size

    double length_;                 // The length of the cylinder
    vector3 origin_;                // The origin of the cylinder in sample space
    vector3 axis_;                  // The axis of the cylinder in sample space
    std::array<double, 10> x_offset_;  // The x offsets as a function of y in cylinder space
    std::array<double, 10> z_offset_;  // The z offsets as a function of y in cylinder space
    std::array<double, 10> radius_;    // The radius as a function of y in cylinder space

    vector3 translation_;           // The translation
    vector3 rotation_vector_;       // The rotation vector (axis and angle)
    vector3 rotation_centre_;       // The centre of rotation

    vector3 A_;                     // The translated start point of the cylinder
    vector3 B_;                     // The translated end point of the cylinder

    double zmin_;                   // The minimum extent of the cylinder
    double zmax_;                   // The maximum extent of the cylinder

    std::array< CylinderMasker::Parameters, 10 > params_; // Interpolation parameters
  };


  /**
   * A class to represent the GRF ice parameters
   */
  class IceParameters {
  public:
    
    double m1;
    double m2;
    double s1;
    double s2;
    double a1;
    double a2;
    double density;

    IceParameters()
      : m1(0),
        m2(1.0/2.88),
        s1(0.731),
        s2(0.081),
        a1(0.199),
        a2(0.801),
        density(0.91) {}
  };

  /**
   * A class to generate a dynamic mask for a Z slice
   */
  class Masker {
  public:

    /**
     * The mask shapes
     */
    enum Shape {
      Cuboid,
      Cylinder
    };

    /**
     * Initialise
     */
    Masker()
     :  shape_(Cuboid) {}

    /**
     * Initialise
     */
    Masker(std::size_t xsize, std::size_t ysize, double pixel_size)
      : shape_(Cuboid) {
      set_image_size(xsize, ysize);
      set_pixel_size(pixel_size);
    }

    /**
     * Get the cuboid masker
     */
    const CuboidMasker& cuboid_masker() const {
      return cuboid_masker_;
    }
    
    /**
     * Get the cylinder masker
     */
    const CylinderMasker& cylinder_masker() const {
      return cylinder_masker_;
    }

    /**
     * Get the x size of the mask image
     */
    std::size_t xsize() const {
      return cuboid_masker_.xsize();
    }

    /**
     * Get the y size of the mask image
     */
    std::size_t ysize() const {
      return cuboid_masker_.ysize();
    }
    
    /**
     * Get the size of the mask image
     */
    std::size_t image_size() const {
      return cuboid_masker_.image_size();
    }

    /**
     * Get the pixel size
     */
    double pixel_size() const {
      return cuboid_masker_.pixel_size();
    }

    /**
     * Get the shape to mask
     */
    Shape shape() const {
      return shape_;
    }

    /**
     * Get the minimum x coordinate
     */
    double xmin() const {
      return shape() == Cuboid
        ? cuboid_masker_.xmin()
        : cylinder_masker_.xmin();
    }
    
    /**
     * Get the minimum y coordinate
     */
    double ymin() const {
      return shape() == Cuboid
        ? cuboid_masker_.ymin()
        : cylinder_masker_.ymin();
    }

    /**
     * Get the minimum z coordinate
     */
    double zmin() const {
      return shape() == Cuboid
        ? cuboid_masker_.zmin()
        : cylinder_masker_.zmin();
    }
    
    /**
     * Get the maximum x coordinate
     */
    double xmax() const {
      return shape() == Cuboid
        ? cuboid_masker_.xmax()
        : cylinder_masker_.xmax();
    }
    
    /**
     * Get the maximum y coordinate
     */
    double ymax() const {
      return shape() == Cuboid
        ? cuboid_masker_.ymax()
        : cylinder_masker_.ymax();
    }

    /**
     * Get the maximum z coordinate
     */
    double zmax() const {
      return shape() == Cuboid
        ? cuboid_masker_.zmax()
        : cylinder_masker_.zmax();
    }
    
    /**
     * Get the rotation origin
     */
    const std::array<double, 3>& rotation_origin() const {
      return cuboid_masker_.rotation_origin();
    }
    
    /**
     * Get the rotation angle
     */
    double rotation_angle() const {
      return cuboid_masker_.rotation_angle();
    }

    /**
     * Get the translation
     */
    const std::array<double, 3>& translation() const {
      return cuboid_masker_.translation();
    }

    /**
     * Get the ice parameters
     */
    const IceParameters& ice_parameters() const {
      return ice_parameters_;
    }

    /**
     * Set the image size
     */
    void set_image_size(std::size_t xsize, std::size_t ysize) {
      MULTEM_ASSERT(xsize > 0);
      MULTEM_ASSERT(ysize > 0);
      cuboid_masker_.set_image_size(xsize, ysize);
      cylinder_masker_.set_image_size(xsize, ysize);
    }
    
    /**
     * Set the pixel size
     */
    void set_pixel_size(double pixel_size) {
      MULTEM_ASSERT(pixel_size > 0);
      cuboid_masker_.set_pixel_size(pixel_size);
      cylinder_masker_.set_pixel_size(pixel_size);
    }

    /**
     * Set a cube mask
     */
    void set_cube(std::array<double,3> offset, double length) {
      MULTEM_ASSERT(length > 0);
      shape_ = Cuboid;
      cuboid_masker_.set_parameters(offset, { length, length, length });
    }
    
    /**
     * Set a cuboid mask
     */
    void set_cuboid(std::array<double,3> offset, std::array<double,3> length) {
      MULTEM_ASSERT(length[0] > 0);
      MULTEM_ASSERT(length[1] > 0);
      MULTEM_ASSERT(length[2] > 0);
      shape_ = Cuboid;
      cuboid_masker_.set_parameters(offset, length);
    }

    /**
     * Set a cylinder mask
     */
    void set_cylinder(
        std::array<double, 3> origin, 
        std::array<double, 3> axis,
        double length,
        std::vector<double> radius,
        std::vector<double> x_offset,
        std::vector<double> z_offset) {
      shape_ = Cylinder;
      cylinder_masker_.set_parameters(origin, axis, length, radius, x_offset, z_offset);
    }

    /**
     * Set a translation along the x axis
     */
    void set_translation(std::array<double, 3> translation) {
      cuboid_masker_.set_translation(translation);
      cylinder_masker_.set_translation(translation);
    }
    
    /**
     * Set a rotation about the x axis
     */
    void set_rotation(std::array<double, 3> centre, std::array<double, 3> rotvec) {
      cuboid_masker_.set_rotation(centre, rotvec);
      cylinder_masker_.set_rotation(centre, rotvec);
    }
   
    /**
     * Compute the mask
     */
    template <typename Iterator>
    void compute(double zs, double ze, Iterator iterator) const {
      if (shape_ == Cuboid) {
        cuboid_masker_.compute_mask(zs, ze, iterator);
      } else if (shape_ == Cylinder) {
        cylinder_masker_.compute_mask(zs, ze, iterator);
      } else {
        MULTEM_ASSERT(false); // Should never reach here
      }
    }

    /**
     * Set the ice parameters
     */
    void set_ice_parameters(const IceParameters &ice_parameters) {
      ice_parameters_ = ice_parameters;
    }

  protected:

    Shape shape_;
    CuboidMasker cuboid_masker_;
    CylinderMasker cylinder_masker_;
    IceParameters ice_parameters_;
  };
}

#endif
