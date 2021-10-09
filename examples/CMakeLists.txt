cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(LarsNDArrayExamples LANGUAGES CXX)

include(../cmake/CPM.cmake)

CPMAddPackage(NAME LarsNDArrayExamples SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

file(GLOB examples CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(example_source ${examples})
  get_filename_component(example_name ${example_source} NAME_WLE)
  add_executable(${example_name} ${example_source})
  set_target_properties(${example_name} PROPERTIES CXX_STANDARD 17)
  target_link_libraries(${example_name} LarsNDArray)
endforeach()