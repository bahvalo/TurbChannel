## TurbChannel
Finite-difference codes for the scale-resolved turbulence simulations (DNS &amp; LES) between parallel plates

## What is this

This repository contains a few finite-different codes for solving incompressible Navier--Stokes equations:

* staggered
* collocated-AB3
* collocated-SRK

Staggered is an implementation of Harlow &amp; Welsh (2-nd order) and Verstappen &amp; Veldman (4-th order) schemes.
For the time integration, the third-order multistep method is used, which is different to the original papers.

Collocated is an implementation of the second-order finite-difference method with both pressure and velocity defined at mesh nodes.
For the time integration, we use the third-order multistep method (collocated-AB3) or the segregated Runge--Kutta methods (collocated-SRK)

All codes are written on CUDA. The calculation can be run on both CPU (with OpenMP) and GPU, but the code must be compiled using CUDA compiler (nvcc) anyway.

CLI11 (https://github.com/CLIUtils/CLI11) is used as a command-line parser

## Requirements

* CUDA toolkit
* FFTW (pressure solver is based on the fast Fourier transform)

## License

This is distributed under the terms of the 3-clause BSD license.

