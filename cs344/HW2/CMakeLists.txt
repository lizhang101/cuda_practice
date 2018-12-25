############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files
cmake_minimum_required(VERSION 3.10)

PROJECT(HW2)
find_package(CUDA 10.0 REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")


enable_language(CUDA)
#INCLUDE_DIRECTORIES(/usr/local/cuda/include /usr/local/include/opencv4)
INCLUDE_DIRECTORIES(/usr/local/cuda/include)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#set (MAKE_CUDA_COMPILER /usr/local/cuda-10.0/bin/nvcc)
set (CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES /usr/local/cuda-10.0/include)
SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW2_files main.cpp reference_calc.cpp compare.cpp)
message("opencv:" ${OpenCV_LIBS})
    
ADD_EXECUTABLE(HW2 ${HW2_files} ${hdr} ${cu})
target_link_libraries(HW2 ${OpenCV_LIBS})
