cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# ---- Project ----

project(LarsNDArray 
  VERSION 0.1
  LANGUAGES CXX
)

if(${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

# ---- Configuration variables ----

option(BUILD_LARS_NDARRAY_EXAMPLES "Enable examples" OFF)
option(ENABLE_LARS_NDARRAY_TESTS "Enable tests" OFF)

# ---- Dependencies ----

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake)

CPMAddPackage(
  NAME LHC
  GIT_REPOSITORY https://github.com/TheLartians/LHC.git
  VERSION 0.3
)

# ---- Minimum CXX ----

if(NOT CMAKE_CXX_STANDARD GREATER 17)
  set(CMAKE_CXX_STANDARD 17)
endif()

# ---- Header target ----

FILE(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/lars/*.h")
add_library(LarsNDArrayHeaders EXCLUDE_FROM_ALL ${headers})
SET_TARGET_PROPERTIES(LarsNDArrayHeaders PROPERTIES LINKER_LANGUAGE CXX)

# ---- Create library ----

add_library(LarsNDArray INTERFACE)

target_link_libraries(LarsNDArray INTERFACE LHC)

target_include_directories(LarsNDArray 
  INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/LarsNDArrayConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# ---- Install ----

install(
  TARGETS LarsNDArray
  EXPORT LarsNDArrayTargets
  LIBRARY DESTINATION lib COMPONENT Runtime
  ARCHIVE DESTINATION lib COMPONENT Development
  RUNTIME DESTINATION bin COMPONENT Runtime
  PUBLIC_HEADER DESTINATION include COMPONENT Development
  BUNDLE DESTINATION bin COMPONENT Runtime
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/LarsNDArrayConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/LarsNDArrayConfig.cmake"
  INSTALL_DESTINATION lib/cmake/LarsNDArray
)

install(
  EXPORT LarsNDArrayTargets 
  DESTINATION lib/cmake/LarsNDArray
)

install(
  FILES 
    "${PROJECT_BINARY_DIR}/LarsNDArrayConfigVersion.cmake"
    "${PROJECT_BINARY_DIR}/LarsNDArrayConfig.cmake"
  DESTINATION 
    lib/cmake/LarsNDArray
)

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION include
)

# ---- Examples ----

IF(${BUILD_LARS_NDARRAY_EXAMPLES})
  add_subdirectory(examples)    
ENDIF()

# ---- Tests ----

if(${ENABLE_LARS_NDARRAY_TESTS})
  ENABLE_TESTING()  
  add_subdirectory(tests)  
endif()

