cmake -S . -B build-xcode -G Xcode \
-DCMAKE_PREFIX_PATH="$(pwd)/third-party/sdl/build;$(pwd)/third-party/metal-cpp/build" \
-DAS_COL_MAJOR=ON -DAS_PRECISION_FLOAT=ON
