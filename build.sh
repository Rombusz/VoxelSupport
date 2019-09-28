#!/bin/bash
echo "Running cmake"
echo $1
cmake . -DCMAKE_BUILD_TYPE=$1
echo "Running make"
make