cmake_minimum_required(VERSION 3.11.0)

# Set the project name
project(multem CXX CUDA)

# Set the cmake module path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Need pybind11 for Python C/C++ extensions
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)
find_package(FFTW REQUIRED)

# Turn off LTO (has problems with cuda)
set(HAS_FLTO False)
set(HAS_FLTO_THIN False)

# Add a C/C++ extension
pybind11_add_module(multem_ext 
  src/multem/multem_ext.cu
  src/multem/multem_ext.cc)

# Ensure we are using C++11
target_compile_features(multem_ext PUBLIC cxx_std_11)

# Set the include directory
target_include_directories(multem_ext PUBLIC 
  src
  MULTEM/src
  MULTEM/gui_multem)

# Link to the CUDA libraries
target_link_libraries(multem_ext PUBLIC
  ${CUDA_LIBRARIES} 
  ${CUDA_CUFFT_LIBRARIES}
  ${FFTW_LIBRARIES})

# Set the coverage compile and link flags for gcc or clang
set(COVERAGE_COMPILE_FLAGS -coverage)
set(COVERAGE_LINK_FLAGS    -coverage)

# Set the coverage compile and link flags in the debug build
#target_compile_options(multem_ext 
#  PUBLIC "$<$<CONFIG:DEBUG>:${COVERAGE_COMPILE_FLAGS}>")
#target_link_options(multem_ext 
#  PUBLIC "$<$<CONFIG:DEBUG>:${COVERAGE_LINK_FLAGS}>")

# Install the python extension
install(TARGETS multem_ext LIBRARY DESTINATION src/multem)
