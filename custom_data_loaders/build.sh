#!/bin/bash

#clang++ -L /usr/lib/x86_64-linux -Wall -fPIC -shared -O3 -std=c++17 -o jack_mouse_test.dll boost_impl.cc -I/usr/include/python3.6m -lboost_system

name=jack_mouse_test

debug=""
#debug="-g"

clang++ -c -fPIC boost_impl.cc -o ${name}.o -Wall -O3 -std=c++17 -I/usr/include/python3.6m $debug

clang++ -L /usr/lib/x86_64-linux-gnu -Wall -fPIC -shared -Wl,-soname,${name}.so -O3 -std=c++17 -o ${name}.cpython-36m-x86_64-linux-gnu.so ${name}.o -I/usr/include/python3.6m -lboost_system -lboost_python3 -lpython3.6m -lboost_numpy3 $debug
