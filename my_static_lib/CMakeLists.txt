cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

project(my_static_lib CXX)

file(
  GLOB_RECURSE STATIC_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS src/*.c*)
add_library(my_static_lib STATIC ${STATIC_SOURCE_FILES})

# header
target_include_directories(my_static_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
