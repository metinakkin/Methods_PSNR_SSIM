cmake_minimum_required(VERSION 2.8)
project( ESPCN )
find_package( OpenCV REQUIRED )
add_executable( ESPCN ESPCN.cpp )
#include_directories( ${OpenCV_INCLUDE_DIRS} )
target_include_directories(ESPCN  PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries( ESPCN ${OpenCV_LIBS} )

message("OpenCV Include Directories: "${OpenCV_INCLUDE_DIRS})
