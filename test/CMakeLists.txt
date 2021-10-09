cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(LarsNDArrayTests LANGUAGES CXX)

# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)

# ---- Dependencies ----

include(../cmake/CPM.cmake)

CPMAddPackage("gh:catchorg/Catch2@2.13.7")
CPMAddPackage("gh:TheLartians/Format.cmake@1.7.0")

if(TEST_INSTALLED_VERSION)
  find_package(LarsNDArray REQUIRED)
else()
  CPMAddPackage(NAME LarsNDArray SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
endif()

# ---- Create binary ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
add_executable(LarsNDArrayTests ${sources})
target_link_libraries(LarsNDArrayTests Catch2::Catch2 LarsNDArray::LarsNDArray)
set_target_properties(LarsNDArrayTests PROPERTIES CXX_STANDARD 17)

# enable compiler warnings
if(NOT TEST_INSTALLED_VERSION)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(LarsNDArray INTERFACE -Wall -Wpedantic -Wextra -Werror)
  elseif(MSVC)
    target_compile_options(LarsNDArray INTERFACE /W4 /WX)
    target_compile_definitions(LarsNDArrayTests PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
  endif()
endif()

# ---- Add LarsNDArrayTests ----

enable_testing()
add_test(LarsNDArrayTests LarsNDArrayTests)

# ---- code coverage ----

if(ENABLE_TEST_COVERAGE)
  target_compile_options(LarsNDArray PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(LarsNDArray PUBLIC -fprofile-arcs -ftest-coverage)
endif()
