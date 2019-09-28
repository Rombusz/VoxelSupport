cmake_minimum_required (VERSION 2.6)
project (IFS_Generation)

find_package( OpenCV REQUIRED )

add_executable(IFS_Support main.cpp)
target_link_libraries( IFS_Support ${OpenCV_LIBS})