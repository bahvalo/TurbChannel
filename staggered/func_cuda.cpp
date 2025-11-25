// ---------------------------------------------------------------------------
// CUDA implementation of the flow solver
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
// Thrust is used for reduction
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#include "config.h"
#include "real3.h"
#include "mytimer.h"
#include "data_global.h"


void GPU_Check_Crash(cudaError_t err, const char *file, int line){
    if(err==cudaSuccess) return;
    printf("\nCUDA error: %s in file %s:%i\n", cudaGetErrorString(err), file, line);
    exit(0);
}
//#define GPU_CHECK_CRASH(X) GPU_Check_Crash(X, __FILE__, __LINE__); // defined in data_global.h

inline void CUFFT_Check_Crash(cufftResult err, const char *file, int line){
    if(err==CUFFT_SUCCESS) return;
    printf("\nCUFFT error: %i in file %s:%i\n", err, file, line);
    exit(0);
}
#define CUFFT_CHECK_CRASH(X) CUFFT_Check_Crash(X, __FILE__, __LINE__);

#define THIS_IS_CUDA
#define __DEVICE__ __device__
#define THRUST_POLICY thrust::device
namespace data_cuda{
    __constant__ __device__ data D;

    template<typename T>
    inline T* alloc(int n){
        T* pntr = NULL;
        GPU_CHECK_CRASH( cudaMalloc( (void**)&pntr, n * sizeof(T) ) );
        return pntr;
    }

    template<typename T>
    inline void dealloc(T*& pntr){
        if(pntr!=NULL) cudaFree(pntr);
        pntr = NULL;
    }

    template<typename T> using thrust_ptr = thrust::device_ptr<T>;
    template<typename T> using thrust_vector = thrust::device_vector<T>;

    template <typename BODY> __global__ static
    void CuKernel1D(BODY body, int N){
        const int k = blockDim.x*blockIdx.x + threadIdx.x;
        if(k >=N) return;
        body(k);
    }

    template <typename BODY> __global__ static
    void CuKernel3D(BODY body){
        int k = (threadIdx.x + blockIdx.x*blockDim.x) + (threadIdx.y + blockIdx.y*blockDim.y)*NX + (threadIdx.z + blockIdx.z*blockDim.z)*NX*NY;
        if(k >= D.N3D) return;
        body(k);
    }

    template <typename DBODY>
    void Forall1D(DBODY &&d_body, int N=data_cpu::D.N3D){
        const int GRID = (N+BLK_1D-1)/BLK_1D;
        CuKernel1D<<<GRID,BLK_1D>>>(d_body,N);
        GPU_CHECK_CRASH(cudaGetLastError());
        #ifdef SYNC_AFTER_EACH_KERNEL
            GPU_CHECK_CRASH(cudaDeviceSynchronize());
        #endif
    }

    template <typename DBODY>
    void Forall(DBODY &&d_body){
        const dim3 GRID(NX/BLK_X, NY/BLK_Y, (data_cpu::D.NZ+BLK_Z-1)/BLK_Z);
        const dim3 BLCK(BLK_X, BLK_Y, BLK_Z);
        CuKernel3D<<<GRID,BLCK>>>(d_body);
        GPU_CHECK_CRASH(cudaGetLastError());
        #ifdef SYNC_AFTER_EACH_KERNEL
            GPU_CHECK_CRASH(cudaDeviceSynchronize());
        #endif
    }

    #include "func_global.h"
}
#undef THRUST_POLICY
#undef __DEVICE__
#undef THIS_IS_CUDA

// Copy data to the constant memory of the device
// Must be in this file, because the constant memory symbol is defined here
void CopyDataToDevice(const data_cuda::data& DATA_CUDA){
    GPU_CHECK_CRASH( cudaMemcpyToSymbol(data_cuda::D, &DATA_CUDA, sizeof(data_cuda::data)) ); // Copy all the data and pointers to the device
}
