cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# ---- Project ----

project(
  LarsNDArray
  VERSION 0.1.1
  LANGUAGES CXX)

if(${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# ---- Dependencies ----

include(cmake/CPM.cmake)

cpmaddpackage(NAME LHC GITHUB_REPOSITORY TheLartians/LHC VERSION 0.8.1)

# CPMAddPackage("gh:TheLartians/LHC@0.4")
cpmaddpackage("gh:TheLartians/PackageProject.cmake@1.8.0")

# ---- Header target ----

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/lars/*.h")
add_library(LarsNDArrayHeaders EXCLUDE_FROM_ALL ${headers})
set_target_properties(LarsNDArrayHeaders PROPERTIES LINKER_LANGUAGE CXX)

# ---- Create library ----

add_library(LarsNDArray INTERFACE)
set_target_properties(LarsNDArray PROPERTIES CXX_STANDARD 17)
target_link_libraries(LarsNDArray INTERFACE LHC)

target_include_directories(
  LarsNDArray
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>)

# ---- Install ----

packageproject(
  NAME
  ${PROJECT_NAME}
  VERSION
  ${PROJECT_VERSION}
  NAMESPACE
  ${PROJECT_NAME}
  BINARY_DIR
  ${PROJECT_BINARY_DIR}
  INCLUDE_DIR
  ${PROJECT_SOURCE_DIR}/include
  INCLUDE_DESTINATION
  include/${PROJECT_NAME}-${PROJECT_VERSION}
  COMPATIBILITY
  SameMajorVersion
  DEPENDENCIES
  "LHC 0.4")
