cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

project(my_header_lib CXX)

add_library(my_header_lib INTERFACE)

# header
target_include_directories(my_header_lib
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

  find_path(BSHOSHANY_THREAD_POOL_INCLUDE_DIRS "BS_thread_pool.hpp")
  target_include_directories(${PROJECT_NAME} INTERFACE ${BSHOSHANY_THREAD_POOL_INCLUDE_DIRS})