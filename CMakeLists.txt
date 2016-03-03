
cmake_minimum_required(VERSION 2.8.3)

GET_FILENAME_COMPONENT(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
PROJECT (${PROJECT_NAME})

SET (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

FILE (GLOB_RECURSE headers ${PROJECT_SOURCE_DIR}/*.h)

add_definitions(-std=c++11)
add_executable(timeit timeit.cpp ${headers})

