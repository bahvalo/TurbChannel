#ifndef CONFIG_H
#define CONFIG_H

// Floating-point arithmetics: 1 for 64-bit floating point variables, 0 for 32-bit
#define USE_DOUBLE_PRECISION 1
#if USE_DOUBLE_PRECISION
    #define real double
#else
    #define real float
#endif

// For debugging or timing
#define SYNC_AFTER_EACH_KERNEL
#define NORMALIZE_PRESSURE_AFTER_EACH_SOLVE

// Turbulent viscosity: 0 for DNS (do not allocate and evaluate nu_t), 1 for LES
#define ENABLE_TURB_VISC 1

// Approximation of the viscous term
enum struct tViscApproximation{ CROSS, NAIVE, GALERKIN };
#define VISC_APPR_TYPE (ENABLE_TURB_VISC ? tViscApproximation::NAIVE : tViscApproximation::CROSS)
//#define VISC_APPR_TYPE tViscApproximation::NAIVE

// Approximation of the convection term
#define CONV_HW 0 // Harlow & Welsh
#define CONV_VV 1 // Verstappen & Veldman
#define CONV_APPR_TYPE CONV_VV


// Mesh dimensions in X and Y. Must be powers of 2
#define NX 64
#define NY 32
#define NXY (NX*NY)
// Maximal number of mesh nodes in Z (for static arrays; also used as the default NZ value). No need to be a power of 2
#define NZ_MAX 32

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
    real visc; // constant viscosity coefficient
    real sour_dpdx; // source term in the X-component of the momentum equation
    real inv_length_scale; // inverted reference length (for turbulence models)

    // Mesh parameters
    real xmin, ymin; // offsets, which do not matter
    real hx, hy, inv_hx, inv_hy, area;
    int NZ, N3D;

    // Constant-memory data (arrays of a small size, which will be copied to the constant memory)
    real ZZ[NZ_MAX+1];
    real DZ[NZ_MAX];

    #if(CONV_APPR_TYPE==CONV_HW)
        real MAT_A[NZ_MAX];
    #endif
    #if(CONV_APPR_TYPE==CONV_VV)
        real DZW[NZ_MAX];
        real GC1[NZ_MAX];
        real GC2[NZ_MAX];
        real GCZ[NZ_MAX];
        real MAT_A[NZ_MAX*3], MAT_D[6];
        real KERG[NZ_MAX], KERG_norm2;
    #endif
};

struct t_cmdline_params{
    int DeviceID = 0; // if there are several GPUs, select one
    int NZ = NZ_MAX; // number of cells in the normal direction
    double TimeMax = 100.; // maximal integration time
    double TimeStartAveraging = 0.; // time where averaging should be started
    int TimeStepsMax = 0x7FFFFFFF; // maximal number of iterations
    double tau = 1e-3; // initial timestep size
    double CFLmin = 0.05; // if CFL falls below this value, then the timestep will be increased
    double CFLmax = 0.5; // if CFL lifts above this value, then the timestep will be decreased
    int SProgress = 200; // number of timesteps between writing to screen and log
    int FProgress = 1000000; // number of timesteps between writing a field
    int PProgress = 100000; // number of timesteps between writing a profile
};
int ReadCmdlineParams(int argc, char** argv, int DeviceCount, t_cmdline_params& PM);

#endif

