cmake_minimum_required(VERSION 2.8)
project( LaPSRN )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( LaPSRN LaPSRN.cpp )
target_link_libraries( LaPSRN ${OpenCV_LIBS} )

