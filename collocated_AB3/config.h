#ifndef CONFIG_H
#define CONFIG_H

// 1 for 64-bit floating point variables, 0 for 32-bit
#define USE_DOUBLE_PRECISION 0
#if USE_DOUBLE_PRECISION
    #define real double
#else
    #define real float
#endif

// For debugging or timing
#define SYNC_AFTER_EACH_KERNEL
#define NORMALIZE_PRESSURE_AFTER_EACH_SOLVE

// Governing equation and numerical method parameters
//#define RE_TAU==180
//#define ENABLE_TURB_VISC 0 // DNS
#define RE_TAU 395
#define ENABLE_TURB_VISC 1  // LES

#define VISC_TERM_GALERKIN ENABLE_TURB_VISC

// Mesh dimensions in X and Y. Must be powers of 2
#define NX 64
#define NY 64
#define NZ_MAX 100 // maximal number of mesh nodes in Z (for static arrays)

// Kernel launch configuration (block size)
// For 1D block grid
#define BLK_1D 256
// For 3D block grid. BLK_X and BLK_Y must be dividers of NX and NY, respectively
#define BLK_X 8
#define BLK_Y 8
#define BLK_Z 4

// Parameters of governing equations, discretizations, etc.
// Must be plain data, no nontrivial constructors allowed
struct main_params{
    // Parameters of governing equations
    int EnableTurbVisc;
    real visc; // constant viscosity coefficient
    real sour_dpdx; // source term in the X-component of the momentum equation
    real inv_length_scale; // inverted reference length (for turbulence models)

    // Mesh parameters
    real xmin, ymin; // offsets, which do not matter
    real hx, hy, inv_hx, inv_hy, area;
    int NZ, N3D;

    // Time discretization parameters
    int StabType; // (0) Du+Sp=0 or (1) Du+Sdp/dt=0

    // Spatial discretization parameters
    int FDmode;
};

#endif

