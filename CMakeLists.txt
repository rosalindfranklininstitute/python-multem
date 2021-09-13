cmake_minimum_required(VERSION 3.17.0)

# Set the project name
project(multem CXX CUDA)

# Set the build type to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set the cuda arch
if(NOT CMAKE_CUDA_ARCHITECTURES)
  if ($ENV{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
  else()
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  endif()
endif()

# Set the cmake module path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Need pybind11 for Python C/C++ extensions
find_package(CUDAToolkit REQUIRED)
find_package(FFTW REQUIRED)

# Add pybind sub directory
add_subdirectory(pybind11)

# Turn off LTO (has problems with cuda)
set(HAS_FLTO False)
set(HAS_FLTO_THIN False)

# Add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH. Required for submission to
# clusters which may not allow export of LD_LIBRARY_PATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH True)

# Use C++11
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Allow Lambdas to be called from device
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Add a C/C++ extension
pybind11_add_module(multem_ext 
  src/multem/multem_ext.cu
  src/multem/multem_ext.cc)

# Not specifying CUDA architecture throws and error.
# Setting this option does not pass arch flag to compiler
set_property(TARGET multem_ext PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# Set the include directory
target_include_directories(multem_ext PUBLIC 
  src
  MULTEM/src)

# Link to the CUDA libraries
target_link_libraries(multem_ext PUBLIC
  CUDA::cudart
  CUDA::cuda_driver
  CUDA::cufft
  ${FFTW_LIBRARIES})

# Install the python extension
install(TARGETS multem_ext LIBRARY DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
