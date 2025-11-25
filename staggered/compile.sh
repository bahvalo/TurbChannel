#!/bin/bash
path_to_fftw="./fftw"
compiler_flags="-x cu --extended-lambda -Xcompiler -fopenmp -Xcompiler -O3 -O2 -arch=native -I$path_to_fftw/api"
linker_flags="-lcufft -lfftw3 -lgomp -L$path_to_fftw/.libs"
nvcc $compiler_flags `ls *.cpp` $linker_flags -o turbch.x
