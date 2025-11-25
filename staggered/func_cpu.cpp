// ---------------------------------------------------------------------------
// CPU implementation of the flow solver
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <omp.h>
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

#define __DEVICE__
#define THRUST_POLICY thrust::host
namespace data_cpu{
    struct data_ext D;

    template<typename T>
    inline T* alloc(int n){
        T* pntr = new T[n];
        if(pntr==NULL){ printf("Memory allocation error\n"); exit(0); }
        return pntr;
    }
    template<typename T>
    inline void dealloc(T*& pntr){
        if(pntr) delete[] pntr;
        pntr = NULL;
    }

    template<typename T> using thrust_ptr = T*;
    template<typename T> using thrust_vector = thrust::host_vector<T>;

    template <typename DBODY>
    void Forall1D(DBODY &&d_body, int N=D.N3D){
        #pragma omp parallel for
        for(int i=0; i<N; i++) d_body(i);
    }

    template <typename DBODY>
    void Forall(DBODY &&d_body){
        #pragma omp parallel for
        for(int i=0; i<D.N3D; i++) d_body(i);
    }

    #include "func_global.h"
}
#undef THRUST_POLICY
#undef __DEVICE__
