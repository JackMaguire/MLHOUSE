#!/bin/bash

clang -O3 -Wall -shared -std=c++14 -fPIC `python3 -m pybind11 --includes` ./test_simple_load.cc -o example`python3-config --extension-suffix`
