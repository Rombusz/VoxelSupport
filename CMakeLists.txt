cmake_minimum_required (VERSION 2.6)
set(CMAKE_CXX_STANDARD 17)

project (IFS_Generation)

find_package(OpenCV REQUIRED)
find_package(PolyVox REQUIRED)
find_package(OpenMesh REQUIRED)


if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "-Wall -pedantic -std=c++17 -fPIC")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif(CMAKE_COMPILER_IS_GNUCXX)

SET( EIGEN3_INCLUDE_DIR "/home/ferenc/BME/Onlab2/ldni/eigen/Eigen/" )
IF( NOT EIGEN3_INCLUDE_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

set(LIBGEOM_DIR "./libgeom") # https://github.com/salvipeter/libgeom
include_directories("${LIBGEOM_DIR}" ${EIGEN3_INCLUDE_DIR})
link_directories("${LIBGEOM_DIR}")

add_library(ldni STATIC ldni/ldni.cc ldni/cell-point.cc)

target_link_libraries(ldni "geom")

add_executable(IFS_Support main.cpp)
target_link_libraries( IFS_Support ${OpenCV_LIBS} ldni libPolyVoxCore.so libPolyVoxUtil.so libOpenMeshCore.so libOpenMeshTools.so) 