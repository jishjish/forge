#!/bin/bash
set -e

mkdir -p third_party/build

clang++ -dynamiclib -std=c++17 \
  -I third_party/metal-cpp \
  -framework Metal \
  -framework Foundation \
  third_party/metal_shim.cpp \
  -o third_party/build/libforge_metal.dylib

echo "Metal shim built successfully"