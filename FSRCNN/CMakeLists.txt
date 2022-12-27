cmake_minimum_required(VERSION 2.8)
project( FSRCNN )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( FSRCNN FSRCNN.cpp )
target_link_libraries( FSRCNN ${OpenCV_LIBS} )

