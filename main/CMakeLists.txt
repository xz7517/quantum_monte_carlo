cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

project(main CXX)

file(
    GLOB_RECURSE MAIN_SOURCE_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS src/*.c*)
add_executable(main ${MAIN_SOURCE_FILES})

# link libraries
target_link_libraries(main PRIVATE my_header_lib my_static_lib)

find_path(BSHOSHANY_THREAD_POOL_INCLUDE_DIRS "BS_thread_pool.hpp")
target_include_directories(main PRIVATE ${BSHOSHANY_THREAD_POOL_INCLUDE_DIRS})

target_compile_options(${PROJECT_NAME} PRIVATE $<IF:$<CONFIG:Release>,-O3 -march=native -ffast-math,-O0 -g3>)
