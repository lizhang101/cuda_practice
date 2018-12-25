############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
cmake_minimum_required(VERSION 3.10)
# collect source files

#PROJECT(HW1 LANGUAGE C CXX CUDA)
PROJECT(HW1)
find_package(CUDA 10.0 REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

enable_language(CUDA)
INCLUDE_DIRECTORIES(/usr/local/cuda/include)

#set (MAKE_CUDA_COMPILER /usr/local/cuda-10.0/bin/nvcc)
set (CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES /usr/local/cuda-10.0/include)

include(CheckLanguage)
check_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW1_files main.cpp reference_calc.cpp compare.cpp)

#CUDA_ADD_EXECUTABLE(HW1 ${HW1_files} ${hdr} ${cu})
ADD_EXECUTABLE(HW1 ${HW1_files} ${hdr} ${cu})
