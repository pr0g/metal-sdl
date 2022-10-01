cmake_minimum_required(VERSION 3.16)

project(metal-sdl LANGUAGES CXX)

find_package(SDL2 REQUIRED CONFIG)

include(FetchContent)

FetchContent_Declare(
  as
  GIT_REPOSITORY https://github.com/pr0g/as.git
  GIT_TAG 9322afdb6dd73aaed256294c2fd1d41ac3d0098b)
FetchContent_Declare(
  as-camera-input
  GIT_REPOSITORY https://github.com/pr0g/as-camera-input.git
  GIT_TAG d47bd06a9dbcfa3f5cb92cdab146d599f0e0b978)

FetchContent_MakeAvailable(as as-camera-input)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE third-party/metal-cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE "-framework Foundation"
          "-framework Quartz"
          "-framework Metal"
          as
          as-camera-input
          SDL2::SDL2
          SDL2::SDL2main)
