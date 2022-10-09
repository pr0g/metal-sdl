cmake_minimum_required(VERSION 3.16)

project(metal-sdl LANGUAGES CXX)

find_package(SDL2 REQUIRED CONFIG)

include(FetchContent)

FetchContent_Declare(
  as
  GIT_REPOSITORY https://github.com/pr0g/as.git
  GIT_TAG 57ac213e5c90363791f59ad29d01d992d7126e9f)
FetchContent_Declare(
  as-camera-input
  GIT_REPOSITORY https://github.com/pr0g/as-camera-input.git
  GIT_TAG 4375a504346305c9dea45c9000a1a06220f8b725)
FetchContent_Declare(
  metal-cpp-cmake
  GIT_REPOSITORY https://github.com/pr0g/metal-cpp-cmake.git
  GIT_TAG 75bcf047702aec5ab08ac90c64694af76ae0910e)

FetchContent_MakeAvailable(as as-camera-input metal-cpp-cmake)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE as as-camera-input SDL2::SDL2
                                              SDL2::SDL2main metal-cpp-cmake)
target_compile_definitions(
  ${PROJECT_NAME}
  PRIVATE $<$<BOOL:${AS_PRECISION_FLOAT}>:AS_PRECISION_FLOAT>
          $<$<BOOL:${AS_PRECISION_DOUBLE}>:AS_PRECISION_DOUBLE>
          $<$<BOOL:${AS_COL_MAJOR}>:AS_COL_MAJOR>
          $<$<BOOL:${AS_ROW_MAJOR}>:AS_ROW_MAJOR>)
