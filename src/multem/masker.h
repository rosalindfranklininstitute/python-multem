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
        rotation_origin_({0, 0, 0}),
        rotation_angle_(0) {
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
      return rotation_origin_;
    }
    
    /**
     * Get the rotation angle
     */
    double rotation_angle() const {
      return rotation_angle_;
    }

    /**
     * Get the translation
     */
    const vector3& translation() const {
      return translation_;
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
    void set_rotation(std::array<double, 3> origin, double angle) {
      rotation_origin_ = origin;
      rotation_angle_ = angle;
      update_geometry();
    }

    /**
     * Update the geometry
     */
    void update_geometry() {
      
      // The extreme points
      double x0 = offset_[0];
      double y0 = offset_[1];
      double z0 = offset_[2];
      double x1 = x0 + size_[0];
      double y1 = y0 + size_[1];
      double z1 = z0 + size_[2];
     
      // The y/z coords of the cuboid
      points_[0] = { y0, z0 };
      points_[1] = { y0, z1 };
      points_[2] = { y1, z1 };
      points_[3] = { y1, z0 };

      // Rotate coords
      detail::rotate2d_around_origin(
          rotation_origin_[1], 
          rotation_origin_[2], 
          rotation_angle_ * M_PI / 180.0,
          points_[0][0], points_[0][1]);
      detail::rotate2d_around_origin(
          rotation_origin_[1], 
          rotation_origin_[2], 
          rotation_angle_ * M_PI / 180.0,
          points_[1][0], points_[1][1]);
      detail::rotate2d_around_origin(
          rotation_origin_[1], 
          rotation_origin_[2], 
          rotation_angle_ * M_PI / 180.0,
          points_[2][0], points_[2][1]);
      detail::rotate2d_around_origin(
          rotation_origin_[1], 
          rotation_origin_[2], 
          rotation_angle_ * M_PI / 180.0,
          points_[3][0], points_[3][1]);

      // Translate the points
      x0 += translation_[0];
      x1 += translation_[0];
      points_[0][0] += translation_[1];
      points_[0][1] += translation_[2];
      points_[1][0] += translation_[1];
      points_[1][1] += translation_[2];
      points_[2][0] += translation_[1];
      points_[2][1] += translation_[2];
      points_[3][0] += translation_[1];
      points_[3][1] += translation_[2];

      // Compute the max and min
      y0 = std::min(
          std::min(points_[0][0], points_[1][0]), 
          std::min(points_[2][0], points_[3][0]));
      y1 = std::max(
          std::max(points_[0][0], points_[1][0]), 
          std::max(points_[2][0], points_[3][0]));
      z0 = std::min(
          std::min(points_[0][1], points_[1][1]), 
          std::min(points_[2][1], points_[3][1]));
      z1 = std::max(
          std::max(points_[0][1], points_[1][1]), 
          std::max(points_[2][1], points_[3][1]));

      // Set the min and max
      pmin_ = { x0, y0, z0 };
      pmax_ = { x1, y1, z1 };
    }
    
    /**
     * Compute the cuboid mask
     */
    template <typename Iterator>
    void compute_mask(double zs, double ze, Iterator iterator) {
      MULTEM_ASSERT(ze > zs);

      // Compute the min and max y
      auto compute_ymin_and_ymax = [](double z, vector2 a, vector2 b, double &y0, double &y1) {
        if (z >= std::min(a[1], b[0]) && z < std::max(a[1], b[1])) {
          double y = (z - a[1]) * (b[0] - a[0]) / (b[1] - a[1]) + a[0];
          y0 = std::min(y0, y);
          y1 = std::max(y0, y);
        }
      };

      // The middle z coordinate of the slice
      double zc = (zs + ze) / 2.0;

      // Only do something if the slice is within range
      if (zs < zmax() && ze > zmin()) {
        
        // Compute min and max y
        double x0 = xmin();
        double x1 = xmax();
				double y1 = ymax();
				double y0 = ymin();
        compute_ymin_and_ymax(zc, points_[0], points_[1], y0, y1);
        compute_ymin_and_ymax(zc, points_[1], points_[2], y0, y1);
        compute_ymin_and_ymax(zc, points_[2], points_[3], y0, y1);
        compute_ymin_and_ymax(zc, points_[3], points_[0], y0, y1);
        MULTEM_ASSERT(y1 >= y0);

        // Convert to pixels
        x0 /= pixel_size_;
        x1 /= pixel_size_;
        y0 /= pixel_size_;
        y1 /= pixel_size_;

        // Compute the slice mask
        for (std::size_t i = 0; i < xsize_; ++i) {
          for (std::size_t j = 0; j < ysize_; ++j) {
            if (i >= x0 && i < x1 && j >= y0 && j < y1) {
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

    std::size_t xsize_;
    std::size_t ysize_;
    double pixel_size_;
    vector3 offset_;
    vector3 size_;
    vector3 translation_;
    vector3 rotation_origin_;
    double rotation_angle_;
    std::array<vector2, 4> points_;
    vector3 pmin_;
    vector3 pmax_;
  };


  /**
   * A cylinder masker
   */
  class CylinderMasker {
  public:

    using vector3 = std::array<double, 3>;
    
    CylinderMasker()
      : xsize_(0),
        ysize_(0),
        pixel_size_(1),
        offset_({0, 0, 0}),
        size_({0, 0, 0}),
        translation_({ 0, 0, 0 }),
        rotation_origin_({0, 0, 0}),
        rotation_angle_(0) {
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
      return rotation_origin_;
    }
    
    /**
     * Get the rotation angle
     */
    double rotation_angle() const {
      return rotation_angle_;
    }

    /**
     * Get the translation
     */
    const vector3& translation() const {
      return translation_;
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
    void set_parameters(std::array<double, 3> offset, double length, double radius) {
      MULTEM_ASSERT(length > 0);
      MULTEM_ASSERT(radius > 0);
      offset_ = offset;
      size_ = { length, 2*radius, 2*radius };
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
    void set_rotation(std::array<double, 3> origin, double angle) {
      rotation_origin_ = origin;
      rotation_angle_ = angle;
      update_geometry();
    }

    /**
     * Update the geometry
     */
    void update_geometry() {
      
      // Compute the centre of the cylinder
      double x0 = offset_[0];
      double y0 = offset_[1];
      double z0 = offset_[2];
      double x1 = x0 + size_[0];
      double y1 = y0 + size_[1];
      double z1 = z0 + size_[2];
      double yc = (y0 + y1) / 2.0;
      double zc = (z0 + z1) / 2.0;

      // Rotate around the origin along the x axis
      detail::rotate2d_around_origin(
          rotation_origin_[1], 
          rotation_origin_[2], 
          rotation_angle_ * M_PI / 180.0,
          yc, zc);

      // translate
      x0 += translation_[0];
      x1 += translation_[0];
      yc += translation_[1];
      zc += translation_[2];

      // Get the new y/z bounds
      y0 = yc - size_[1] / 2.0;
      y1 = yc + size_[1] / 2.0;
      z0 = zc - size_[2] / 2.0;
      z1 = zc + size_[2] / 2.0;

      // Set the min and max
      pmin_ = { x0, y0, z0 };
      pmax_ = { x1, y1, z1 };
    }
    
    /**
     * Compute the cylinder mask
     */
    template <typename Iterator>
    void compute_mask(double zs, double ze, Iterator iterator) {
      MULTEM_ASSERT(ze > zs);

      // Convert to pixels
      double x0 = xmin() / pixel_size_;
      double x1 = xmax() / pixel_size_;
      double y0 = ymin() / pixel_size_;
      double y1 = ymax() / pixel_size_;
      double z0 = zmin() / pixel_size_;
      double z1 = zmax() / pixel_size_;
      double yc = (y0 + y1) / 2.0;
      double zc = (z0 + z1) / 2.0;
      zs /= pixel_size_;
      ze /= pixel_size_;

      // Compute the radius squared
      double radius2 = std::pow(0.5 * size_[1] / pixel_size_, 2);

      // Compute the slice mask
      if (zs < z1 && ze > z0) {
        for (std::size_t i = 0; i < xsize_; ++i) {
          for (std::size_t j = 0; j < ysize_; ++j) {
            double r1 = (j+0.5-yc)*(j+0.5-yc)+(zs-zc)*(zs-zc);
            double r2 = (j+0.5-yc)*(j+0.5-yc)+(ze-zc)*(ze-zc);
            if (i >= x0 && i < x1 && std::min(r1, r2) < radius2) {
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

    std::size_t xsize_;
    std::size_t ysize_;
    double pixel_size_;
    vector3 offset_;
    vector3 size_;
    vector3 translation_;
    vector3 rotation_origin_;
    double rotation_angle_;
    vector3 pmin_;
    vector3 pmax_;

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
     * Get the offset
     */
    const std::array<double, 3>& offset() const {
      return shape() == Cuboid
        ? cuboid_masker_.offset()
        : cylinder_masker_.offset();
    }

    /**
     * Get the size
     */
    const std::array<double, 3>& size() const {
      return shape() == Cuboid
        ? cuboid_masker_.size()
        : cylinder_masker_.size();
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
    void set_cylinder(std::array<double, 3> offset, double length, double radius) {
      MULTEM_ASSERT(length > 0);
      MULTEM_ASSERT(radius > 0);
      shape_ = Cylinder;
      cylinder_masker_.set_parameters(offset, length, radius);
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
    void set_rotation(std::array<double, 3> origin, double angle) {
      cuboid_masker_.set_rotation(origin, angle);
      cylinder_masker_.set_rotation(origin, angle);
    }
   
    /**
     * Compute the mask
     */
    template <typename Iterator>
    void compute(double zs, double ze, Iterator iterator) {
      if (shape_ == Cuboid) {
        cuboid_masker_.compute_mask(zs, ze, iterator);
      } else if (shape_ == Cylinder) {
        cylinder_masker_.compute_mask(zs, ze, iterator);
      } else {
        MULTEM_ASSERT(false); // Should never reach here
      }
    }


  protected:

    Shape shape_;
    CuboidMasker cuboid_masker_;
    CylinderMasker cylinder_masker_;
  };
}

#endif
