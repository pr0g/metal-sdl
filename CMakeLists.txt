cmake_minimum_required(VERSION 3.16)

project(metal-sdl LANGUAGES CXX)

find_package(SDL2 REQUIRED CONFIG)

include(FetchContent)

FetchContent_Declare(
  as
  GIT_REPOSITORY https://github.com/pr0g/as.git
  GIT_TAG a08c946ecc477e38beff41aa0e47849a8c0383c6)
FetchContent_Declare(
  as-camera-input-sdl
  GIT_REPOSITORY https://github.com/pr0g/as-camera-input-sdl.git
  GIT_TAG a2c53f7ef12cea081b632d559477f1549055a080)
FetchContent_Declare(
  metal-cpp-cmake
  GIT_REPOSITORY https://github.com/pr0g/metal-cpp-cmake.git
  GIT_TAG 6f6bfd1dfae7465e004a552cedeee7f3f4ef1fe2)

FetchContent_MakeAvailable(as as-camera-input-sdl metal-cpp-cmake)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE as as-camera-input-sdl SDL2::SDL2
                                              SDL2::SDL2main metal-cpp-cmake)
target_compile_definitions(
  ${PROJECT_NAME}
  PRIVATE $<$<BOOL:${AS_PRECISION_FLOAT}>:AS_PRECISION_FLOAT>
          $<$<BOOL:${AS_PRECISION_DOUBLE}>:AS_PRECISION_DOUBLE>
          $<$<BOOL:${AS_COL_MAJOR}>:AS_COL_MAJOR>
          $<$<BOOL:${AS_ROW_MAJOR}>:AS_ROW_MAJOR>)
