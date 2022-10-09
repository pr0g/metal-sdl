cmake_minimum_required(VERSION 3.15)

project(metal-cpp)

include(ExternalProject)

ExternalProject_Add(
  metal-cpp-cmake
  GIT_REPOSITORY https://github.com/pr0g/metal-cpp-cmake.git
  GIT_TAG a5ef08d590d3ad35c818fe2a43911a15d4f33c25
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/metal-cpp/build
  INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
  CMAKE_ARGS -D CMAKE_INSTALL_PREFIX=<INSTALL_DIR>)
