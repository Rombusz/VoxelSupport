cmake_minimum_required (VERSION 2.6)
set(CMAKE_CXX_STANDARD 14)

project (IFS_Generation)

find_package(OpenCV REQUIRED)
find_package(PolyVox REQUIRED)
find_package(OpenMesh REQUIRED)

add_executable(IFS_Support main.cpp)
target_link_libraries( IFS_Support ${OpenCV_LIBS} libPolyVoxCore.so libPolyVoxUtil.so libOpenMeshCore.so libOpenMeshTools.so) 