#ifndef DATA_GLOBAL_H
#define DATA_GLOBAL_H

// these headers must included to include data_global_inc.h
#include <cufft.h>
#include <fftw3.h>
#include "config.h"
#include "real3.h"

namespace data_cpu{
    #define __DEVICE__
    #include "data_global_inc.h"
    #undef __DEVICE__
}

void GPU_Check_Crash(cudaError_t err, const char *file, int line);
#define GPU_CHECK_CRASH(X) GPU_Check_Crash(X, __FILE__, __LINE__);

namespace data_cuda{
    #define THIS_IS_CUDA
    #define __DEVICE__ __device__
    #include "data_global_inc.h"
    #undef __DEVICE__
    #undef THIS_IS_CUDA
}

// Additional methods for the CPU-only code
struct data_ext : data_cpu::data{
    void FillMatrix(real* M); // 7-diagonal pressure matrix (for the constant-in-xy mode)
    void InitDerivativeMeshData();
    real3 GetCoor(int in, int ivar) const;

    // Input & output
    void DumpData(const char* fname, const real3* velocity, const real* pressure, const real* qressure) const;
    int ReadData(const char* fname, real3* velocity, real* pressure) const;
    template<typename fpv> void DumpProfiles(const real9<fpv>* av, const char* fname);
    void SetInitialFields(int current_field, double u_tau);

    // Self-test
    void CheckConvectionConsistency(int print_all);
    void CheckGradientKernel();
    void CheckPressureMatrix();
    void CheckPressureSolver();
    void CheckConservation(int itest);
    void CheckViscNaiveConsistency(int IsMeshUniform, int print_all);
    //void CheckViscGalerkinConsistency(int IsMeshUniform, int print_all);
    void MakeCPUChecks(int IsMeshUniform);

    // Check that CPU and GPU give the same results
    void CheckGPU_ApplyGradient(data_cuda::data& G);
    void CheckGPU_UnapplyGradientAndExplicit(data_cuda::data& G);
    void CheckGPU_CalcProfiles(data_cuda::data& G);
    void MakeCPUGPUChecks(data_cuda::data& G);
};

namespace data_cpu{
    extern struct data_ext D;
}

#endif
