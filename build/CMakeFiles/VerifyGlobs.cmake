# CMAKE generated file: DO NOT EDIT!
# Generated by CMake Version 3.22
cmake_policy(SET CMP0009 NEW)

# MAIN_SOURCE_FILES at main/CMakeLists.txt:5 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false RELATIVE "/home/zx/qmc-annealer/main" "/home/zx/qmc-annealer/main/src/*.c*")
set(OLD_GLOB
  "src/main.cpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/zx/qmc-annealer/build/CMakeFiles/cmake.verify_globs")
endif()

# STATIC_SOURCE_FILES at my_static_lib/CMakeLists.txt:5 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false RELATIVE "/home/zx/qmc-annealer/my_static_lib" "/home/zx/qmc-annealer/my_static_lib/src/*.c*")
set(OLD_GLOB
  "src/my_static_lib.cpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/zx/qmc-annealer/build/CMakeFiles/cmake.verify_globs")
endif()

# TEST_SOURCE_FILES at test/CMakeLists.txt:8 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false RELATIVE "/home/zx/qmc-annealer/test" "/home/zx/qmc-annealer/test/src/*.c*")
set(OLD_GLOB
  "src/my_header_lib_test.cpp"
  "src/my_static_lib_test.cpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/zx/qmc-annealer/build/CMakeFiles/cmake.verify_globs")
endif()
