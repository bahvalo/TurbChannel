#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <omp.h>
// FFT
#include <cufft.h>
#include <fftw3.h>
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
#include "io.h"
#include "mytimer.h"

// Floating-point constants
#define C1_90   real(0.01111111111111111111)
#define C1_60   real(0.01666666666666666666)
#define C1_36   real(0.02777777777777777777)
#define C1_18   real(0.05555555555555555555)
#define C1_12   real(0.08333333333333333333)
#define C1_9    real(0.11111111111111111111)
#define C1_6    real(0.16666666666666666666)
#define C1_3    real(0.33333333333333333333)
#define C2_3    real(0.66666666666666666666)
#define C4_3    real(1.33333333333333333333)
#define C49_18  real(2.72222222222222222222)
#define Pi      real(3.14159265358979323846)

// Timers used in methods of the data structure -- should be defined before including data_global.h
MyTimer t_ExplicitTerm, t_ApplyGradient, t_PressureSolver[4];


// ---------------------------------------------------------------------------
// CPU implementation of the flow solver
// ---------------------------------------------------------------------------

inline void CPU_Check_Crash(int err, const char *file, int line){
    if(!err) return;
    printf("\nCPU error in file %s:%i\n", file, line);
    exit(0);
}
#define CPU_CHECK_CRASH(X) CPU_Check_Crash(X, __FILE__, __LINE__);

#define __DEVICE__
#define __CONSTANT__
#define THRUST_POLICY thrust::host
namespace data_cpu{
    template<typename T>
    inline T* alloc(int n){
        T* pntr = new T[n];
        CPU_CHECK_CRASH(pntr==NULL);
        return pntr;
    }
    template<typename T>
    inline void dealloc(T*& pntr){
        if(pntr) delete[] pntr;
        pntr = NULL;
    }

    template<typename T> using thrust_ptr = T*;
    template<typename T> using thrust_vector = thrust::host_vector<T>;

    #include "data_global.h"

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
#undef __CONSTANT__


// ---------------------------------------------------------------------------
// CUDA implementation of the flow solver
// ---------------------------------------------------------------------------

inline void GPU_Check_Crash(cudaError_t err, const char *file, int line){
    if(err==cudaSuccess) return;
    printf("\nCUDA error: %s in file %s:%i\n", cudaGetErrorString(err), file, line);
    exit(0);
}
#define GPU_CHECK_CRASH(X) GPU_Check_Crash(X, __FILE__, __LINE__);

inline void CUFFT_Check_Crash(cufftResult err, const char *file, int line){
    if(err==CUFFT_SUCCESS) return;
    printf("\nCUFFT error: %i in file %s:%i\n", err, file, line);
    exit(0);
}
#define CUFFT_CHECK_CRASH(X) CUFFT_Check_Crash(X, __FILE__, __LINE__);

#define THIS_IS_CUDA
#define __DEVICE__ __device__
#define __CONSTANT__ __constant__ __device__
#define THRUST_POLICY thrust::device
namespace data_cuda{
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

    #include "data_global.h"

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
#undef __CONSTANT__
#undef THIS_IS_CUDA


// ---------------------------------------------------------------------------
// CPU-only code
// ---------------------------------------------------------------------------

int IsPowerOfTwo(int i){
    if(i<=0) return 0;
    while(i>1){ if(i&1) return 0; i>>=1; };
    return 1;
}

int main(int argc, char** argv){
    // Command-line parameters
    int DeviceCount;
    GPU_CHECK_CRASH( cudaGetDeviceCount(&DeviceCount) );
    t_cmdline_params PM;
    if(ReadCmdlineParams(argc, argv, DeviceCount, PM)) exit(0);
    // Checks
    printf("sizeof(data_cuda::data) = %lu\n", sizeof(data_cuda::data));
    if(!IsPowerOfTwo(NX) || !IsPowerOfTwo(NY) || NX%BLK_X || NY%BLK_Y) { printf("WRONG DEFINES\n"); exit(0); }
    if(PM.DeviceID<0 || PM.DeviceID>=DeviceCount) { printf("DEVICE ID OUT OF RANGE\n"); exit(0); }
    // Problem parameters
    const double Re_tau = RE_TAU; // u_tau*H/visc
    const double H = 1.; // channel half-height
    const double visc = 1./Re_tau; // kinematic viscosity (affects the velocity scale only)
    const double u_tau = Re_tau*visc; // expected friction velocity
    const double sour_dpdx = u_tau*u_tau / H; // source in the momentum equation per unit volume
    const double Hx = 4.*Pi*H, Hy = 4.*Pi*H/3, Hz = 2.*H; // domain size
    const double xmin = 0., ymin = -0.5*Hy, zmin = 0.; // offset, does not matter

    std::vector<double> Z_plus; // mesh coordinates in normal direction ("y+")
    #if RE_TAU==180
    {
        #define Z Z_plus
        const double y1 = 50.; //0.49; // first step
        const double ybulk = 2.2; // max step
        const double coeff = 1.05; // geometric progression coefficient

        Z.push_back(0.);
        Z.push_back(y1);
        while(1){
            double hz = (Z[Z.size()-1]-Z[Z.size()-2]) * coeff;
            if(hz>=ybulk) { printf("Mesh generation: max step reached at y+ = %.2f\n", Z[Z.size()-1]); break; }
            Z.push_back(Z[Z.size()-1] + hz);
        };
        int n = int((Re_tau - Z[Z.size()-1]) / ybulk) + 1;
        double hz_bulk = (Re_tau - Z[Z.size()-1])/n;
        for(int i=0; i<n-1; i++) Z.push_back(Z[Z.size()-1] + hz_bulk);
        Z.push_back(Re_tau);
        for(int i= (int)Z.size()-2; i>=0; i--) Z.push_back(2.*Re_tau-Z[i]);
        #undef Z
    }
    #else
    {
        Z_plus = std::vector<double>({
            0.000000000000e+000, 1.000000000000e-003, 2.112480000000e-003, 3.350080000000e-003, 4.726890000000e-003, 6.258560000000e-003, 7.962510000000e-003, 9.858110000000e-003, 1.196690000000e-002, 1.431290000000e-002, 1.692280000000e-002, 1.982630000000e-002, 2.305630000000e-002, 2.664960000000e-002, 3.064710000000e-002, 3.509420000000e-002, 4.004150000000e-002, 4.554520000000e-002, 5.166810000000e-002, 5.847960000000e-002, 6.605720000000e-002, 7.448710000000e-002, 8.386530000000e-002, 9.429820000000e-002, 1.059050000000e-001, 1.188170000000e-001, 1.331810000000e-001, 1.491610000000e-001, 1.669380000000e-001, 1.867150000000e-001, 2.087160000000e-001, 2.331920000000e-001, 2.604200000000e-001, 2.907120000000e-001, 3.244100000000e-001, 3.618990000000e-001, 4.036050000000e-001, 4.500010000000e-001, 4.999990000000e-001, 5.499990000000e-001, 5.963950000000e-001, 6.381010000000e-001, 6.755900000000e-001, 7.092880000000e-001, 7.395800000000e-001, 7.668080000000e-001, 7.912840000000e-001, 8.132850000000e-001, 8.330620000000e-001, 8.508390000000e-001, 8.668190000000e-001, 8.811830000000e-001, 8.940950000000e-001, 9.057020000000e-001, 9.161350000000e-001, 9.255130000000e-001, 9.339430000000e-001, 9.415200000000e-001, 9.483320000000e-001, 9.544550000000e-001, 9.599590000000e-001, 9.649060000000e-001, 9.693530000000e-001, 9.733500000000e-001, 9.769440000000e-001, 9.801740000000e-001, 9.830770000000e-001, 9.856870000000e-001, 9.880330000000e-001, 9.901420000000e-001, 9.920370000000e-001, 9.937410000000e-001, 9.952730000000e-001, 9.966500000000e-001, 9.978880000000e-001, 9.990000000000e-001, 1.000000000000e+000
        });
        for(unsigned int i=0; i<Z_plus.size(); i++) Z_plus[i] *= Hz/visc;
        //for(unsigned int i=0; i<Z_plus.size(); i++) Z_plus[i] = double(i)/(Z_plus.size()-1) * Hz/visc;
    }
    #endif
    const int NZ = Z_plus.size();
    if(NZ > NZ_MAX) { printf("WRONG NZ_MAX=%i (actual number of nodes = %i)\n", NZ_MAX, NZ); exit(0); }
    for(int i=0; i<NZ; i++) data_cpu::ZZ[i] = Z_plus[i]*visc;
    for(int i=0; i<NZ-1; i++) data_cpu::DZ[i] = data_cpu::ZZ[i+1]-data_cpu::ZZ[i];
    const int N3D = NX*NY*NZ;

    {
        main_params& MD = data_cpu::D;
        MD.StabType = 1;

        // Other parameters
        MD.visc = visc;
        MD.sour_dpdx = sour_dpdx;
        MD.inv_length_scale = u_tau/visc;
        MD.EnableTurbVisc = ENABLE_TURB_VISC;

        // Mesh parameters
        MD.NZ = NZ;
        MD.N3D = NX*NY*NZ;
        MD.hx = Hx/NX;
        MD.hy = Hy/NY;
        MD.inv_hx = 1./MD.hx;
        MD.inv_hy = 1./MD.hy;
        MD.area = Hx*Hy*(data_cpu::ZZ[NZ-1]-data_cpu::ZZ[0]);
        MD.xmin = xmin;
        MD.ymin = ymin;

        // Spatial discretization
        MD.FDmode = 1;
    }

    #define Z_CPU data_cpu::ZZ
    #define DATA_CPU data_cpu::D

    #if VISC_TERM_GALERKIN==0
        if(DATA_CPU.EnableTurbVisc){ printf("VISC_TERM_GALERKIN reqiured for a turbulence model\n"); exit(0); }
    #endif

    // Data on host with device pointers
    struct data_cuda::data DATA_CUDA;
    // Copy the main parameters to DATA_CUDA
    *(static_cast<main_params*>(&DATA_CUDA)) = *(static_cast<main_params*>(&DATA_CPU));

    // Alloc global memory
    DATA_CPU.alloc_all();
    DATA_CPU.InitFourier();

    // Preparing the device
    GPU_CHECK_CRASH( cudaSetDevice(PM.DeviceID) ); // Set the device
    DATA_CUDA.alloc_all(); // alloc memory on device

    // Send the structure to DATA_ON_DEVICE
    GPU_CHECK_CRASH( cudaMemcpyToSymbol(data_cuda::D, &DATA_CUDA, sizeof(data_cuda::data)) );
    DATA_CUDA.InitFourier();
    GPU_CHECK_CRASH( cudaMemcpyToSymbol(data_cuda::D, &DATA_CUDA, sizeof(data_cuda::data)) );

    GPU_CHECK_CRASH( cudaMemcpyToSymbol(data_cuda::DZ, data_cpu::DZ, sizeof(double)*NZ) );
    GPU_CHECK_CRASH( cudaMemcpyToSymbol(data_cuda::ZZ, data_cpu::ZZ, sizeof(double)*NZ) );

    {
        double* A = data_cpu::MAT_A;
        A[0]=0.;
        for(int j=1; j<NZ; j++) A[j]=DATA_CPU.hx*DATA_CPU.hy/data_cpu::DZ[j-1];
        if(DATA_CPU.FDmode){ A[1]*=0.5; A[NZ-1]*=0.5; }
    }
    GPU_CHECK_CRASH( cudaMemcpyToSymbol(data_cuda::MAT_A, data_cpu::MAT_A, sizeof(double)*NZ) );

    // Comparison of specific subroutines on CPU and GPU
    #if 1
    if(1){
        for(int in=0; in<N3D; in++) DATA_CPU.pres[0][in] = sin(double(in)*in);
        GPU_CHECK_CRASH( cudaMemcpy( DATA_CUDA.pres[0], DATA_CPU.pres[0], N3D*sizeof(real), cudaMemcpyHostToDevice ) );
        DATA_CUDA.ApplyGradient(DATA_CUDA.pres[0], DATA_CUDA.vel[0]);
        DATA_CPU.ApplyGradient(DATA_CPU.pres[0], DATA_CPU.vel[0]);
        GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.vel[1], DATA_CUDA.vel[0], N3D*sizeof(real3), cudaMemcpyDeviceToHost ) );
        cudaDeviceSynchronize();
        double err = 0.;
        for(int in=0; in<N3D; in++) err += abs(DATA_CPU.vel[0][in]-DATA_CPU.vel[1][in]);
        printf("Error in ApplyGradient: %e\n", err);
    }
    if(1){
        for(int in=0; in<N3D; ++in){
            int i[3] = {in%NX, (in/NX)%NY, in/(NX*NY)};
            double x = i[0]*DATA_CPU.hx, /*y = i[1]*DATA_CPU.hy,*/ z = Z_plus[i[2]]*visc;
            double zplus = std::min(Z_plus[i[2]], Re_tau*2-Z_plus[i[2]]);
            DATA_CPU.vel[0][in][0] = 0.5*(z-zmin)*(zmin+Hz-z)*sour_dpdx/visc;
            DATA_CPU.vel[0][in][1] = 1e-1*sin(0.5*x)*zplus*exp(-0.01*zplus*zplus);
            DATA_CPU.vel[0][in][2] = 0.;
            DATA_CPU.nu_t[in] = 0.;
            for(int idir=0; idir<3; idir++) if(i[2]!=0 && i[2]!=NZ-1) DATA_CPU.vel[0][in][idir] += 1e-1*sin(double(in)*in);
        }
        GPU_CHECK_CRASH( cudaMemcpy( DATA_CUDA.nu_t, DATA_CPU.nu_t, N3D*sizeof(real), cudaMemcpyHostToDevice ) );
        GPU_CHECK_CRASH( cudaMemcpy( DATA_CUDA.vel[0], DATA_CPU.vel[0], N3D*sizeof(real3), cudaMemcpyHostToDevice ) );
        DATA_CUDA.UnapplyGradient(DATA_CUDA.vel[0], DATA_CUDA.pres[0]);
        DATA_CPU.UnapplyGradient(DATA_CPU.vel[0], DATA_CPU.pres[0]);
        GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.pres[1], DATA_CUDA.pres[0], N3D*sizeof(real), cudaMemcpyDeviceToHost ) );
        cudaDeviceSynchronize();
        double err = 0.;
        for(int in=0; in<N3D; in++) err += fabs(DATA_CPU.pres[0][in]-DATA_CPU.pres[1][in]);
        printf("Error in UnapplyGradient: %e\n", err);

        DATA_CUDA.ExplicitTerm(0., DATA_CUDA.vel[0], DATA_CUDA.pres[0], DATA_CUDA.vel[1]);
        DATA_CPU.ExplicitTerm(0., DATA_CPU.vel[0], DATA_CPU.pres[0], DATA_CPU.vel[1]);
        GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.vel[2], DATA_CUDA.vel[1], N3D*sizeof(real3), cudaMemcpyDeviceToHost ) );
        cudaDeviceSynchronize();
        err = 0.;
        for(int in=0; in<N3D; in++) err += abs(DATA_CPU.vel[1][in]-DATA_CPU.vel[2][in]);
        printf("Error in ExplicitTerm: %e\n", err);
    }
    #endif

    // Initial data
    int current_field = 0;
    for(int in=0; in<N3D; ++in){
        int i[3] = {in%NX, (in/NX)%NY, in/(NX*NY)};
        double x = i[0]*DATA_CPU.hx, y = i[1]*DATA_CPU.hy, z = Z_plus[i[2]]*visc;
        double zplus = std::min(Z_plus[i[2]], Re_tau*2-Z_plus[i[2]]);
        real3 vel;
        vel[0] = 0.5*(z-zmin)*(zmin+Hz-z)*sour_dpdx/visc;
        //vel[1] = sin(x)*zplus*exp(-0.01*zplus*zplus);
        //vel[2] = 0.;
        //for(int idir=0; idir<3; idir++) if(i[2]!=0 && i[2]!=NZ-1) vel[idir] += 1e-1*double(rand())/RAND_MAX;

        //vel[2] = 0.3*sin(x)*(1-cos(2.*Pi*z/Hz));
        //vel[0]+= 0.3*cos(x)*(2*Pi/Hz)*sin(2.*Pi*z/Hz);

        vel[1] += sin(x)*cos(2.*Pi*y/Hy)*zplus*exp(-0.01*zplus*zplus)/(2.*Pi/Hy);
        vel[0] -= cos(x)*sin(2.*Pi*y/Hy)*zplus*exp(-0.01*zplus*zplus);


        DATA_CPU.vel[current_field][in]=vel;
        DATA_CPU.pres[current_field][in]=0.;
    }

    int read_result = ReadData(DATA_CPU, "input.vtk", DATA_CPU.vel[current_field], DATA_CPU.pres[current_field]);
    if(read_result) printf("input.vtk not found. Starting from the beginning\n");
    else printf("Initial data read\n");

    #define WORK_ON_GPU // uncomment for CUDA solver

    #ifdef WORK_ON_GPU
        #define FlowSolver DATA_CUDA
    #else
        #define FlowSolver DATA_CPU
        omp_set_num_threads(2);
    #endif

    #ifdef WORK_ON_GPU
    for(int i=0; i<4; i++){
        GPU_CHECK_CRASH( cudaMemcpy( DATA_CUDA.vel[i], DATA_CPU.vel[i], N3D*sizeof(real3), cudaMemcpyHostToDevice ) );
        GPU_CHECK_CRASH( cudaMemcpy( DATA_CUDA.pres[i], DATA_CPU.pres[i], N3D*sizeof(real), cudaMemcpyHostToDevice ) );
    }
    cudaDeviceSynchronize(); // for the correct timing
    #endif

    // Time integration
    printf("Start of time integration (N=%i,%i,%i)\n", NX, NY, NZ);
    FILE* FILE_LOG = fopen("log.dat", "wt");
    int iTimeStep=0, isFinalStep=0;
    real3 FlowRate_prev = FlowSolver.CalcIntegral(FlowSolver.vel[current_field])/DATA_CPU.area;

    std::vector<real9> intergal_av_u(NZ);
    double intergal_time = 0.; // time of averaging

    double t = 0.; // current time
    double tau = PM.tau; // initial value of the timestep size
    const int max_bdf_order = 3;
    int iTimeStep_start = 0; // first timestep after changed tau

    MyTimer t_Total, t_Step, t_velprof, t_reduction, t_velprof_copy, t_output;
    t_Total.beg();
    for(iTimeStep=iTimeStep_start; !isFinalStep; iTimeStep++){
        t_Step.beg();
        FlowSolver.Step(iTimeStep*tau, tau, current_field, std::min(iTimeStep-iTimeStep_start+1,max_bdf_order));
        current_field = (current_field+1)%4;
        t_Step.end();

        t_reduction.beg();
        real3 FlowRate = FlowSolver.CalcIntegral(FlowSolver.vel[current_field])/DATA_CPU.area;
        t_reduction.end();
        t_velprof.beg();
        FlowSolver.CalcVelocityProfile(FlowSolver.vel[current_field]);
        t_velprof.end();
        #ifdef WORK_ON_GPU
            t_velprof_copy.beg();
            GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.av_u, DATA_CUDA.av_u, NZ*sizeof(real9), cudaMemcpyDeviceToHost ) );
            t_velprof_copy.end();
        #endif
        if(t>=PM.TimeStartAveraging){
            for(int iz=0; iz<NZ; iz++) intergal_av_u[iz] += tau*DATA_CPU.av_u[iz];
            intergal_time += tau;
        }

        t_output.beg();
        t += tau;
        if(t>=PM.TimeMax || iTimeStep+1==PM.TimeStepsMax) isFinalStep=1;
        double dudy_current = (DATA_CPU.av_u[1].u[0]-0.)/data_cpu::DZ[0];
        double u_fric_current = dudy_current<0. ? 0. : sqrt(dudy_current*visc);
        double dudy_integral = intergal_time>0 ? intergal_av_u[1].u[0]/(data_cpu::DZ[0]*intergal_time) : 0.;
        double u_fric_integral = dudy_integral<0. ? 0. : sqrt(dudy_integral*visc);

        int DoPrintStdout = (iTimeStep%PM.SProgress==0) || isFinalStep;
        int DoPrintFile   = (iTimeStep%PM.SProgress==0) || isFinalStep;
        double CFL=FlowSolver.CalcCFL(FlowSolver.vel[current_field], tau, 3 /*Both conv and visc*/);
        if(DoPrintStdout) printf("T=%f Ubulk=%.04f tau=%.1e CFL=%.04f Ufric=%.04f UfricAv=%.04f\n", t, FlowRate[0], tau, CFL, u_fric_current, u_fric_integral);
        if(DoPrintFile) fprintf(FILE_LOG, "%f %f %e %f %f %f\n", t, FlowRate[0], tau, CFL, u_fric_current, u_fric_integral);

        int DoWriteOutput1 = (iTimeStep!=0 && iTimeStep%PM.FProgress==0);
        int DoWriteOutput2 = (iTimeStep!=0 && iTimeStep%PM.PProgress==0);
        if(DoWriteOutput1){
            char fname[256];
            sprintf(fname, "q%05i.vtk", iTimeStep);
            #ifdef WORK_ON_GPU
            GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.vel[current_field], DATA_CUDA.vel[current_field], N3D*sizeof(real3), cudaMemcpyDeviceToHost ) );
            GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.pres[current_field], DATA_CUDA.pres[current_field], N3D*sizeof(real), cudaMemcpyDeviceToHost ) );
            #endif
            DumpData(DATA_CPU, Z_CPU, fname, DATA_CPU.vel[current_field], DATA_CPU.pres[current_field], NULL);
        }
        if(DoWriteOutput2){
            char fname[256];
            sprintf(fname, "r%05i.dat", iTimeStep);
            DumpProfiles(DATA_CPU, Z_CPU, DATA_CPU.av_u, visc, u_tau, fname);
        }
        t_output.end();
        FlowRate_prev = FlowRate;

        while(CFL>PM.CFLmax){
            CFL *= 0.5;
            tau *= 0.5;
            iTimeStep_start = iTimeStep+1;
        }
        if(CFL<PM.CFLmin){
            CFL *= 1.5;
            tau *= 1.5;
            iTimeStep_start = iTimeStep+1;
        }
    }
    fclose(FILE_LOG);

    cudaDeviceSynchronize(); // for the correct timing
    t_Total.end();
    printf("Time: total %.04f, step %.04f, expl %.04f, grad %.04f, pres %.04f (F=%.04f T=%.04f N=%.04f), velprof %.04f (copy %.04f), reduction %.04f, output %.04f\n", t_Total.timer, t_Step.timer, t_ExplicitTerm.timer, t_ApplyGradient.timer, t_PressureSolver[0].timer, t_PressureSolver[1].timer, t_PressureSolver[2].timer, t_PressureSolver[3].timer, t_velprof.timer, t_velprof_copy.timer, t_reduction.timer, t_output.timer);

    #ifdef WORK_ON_GPU
    GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.vel[current_field], DATA_CUDA.vel[current_field], N3D*sizeof(real3), cudaMemcpyDeviceToHost ) );
    GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.pres[current_field], DATA_CUDA.pres[current_field], N3D*sizeof(real), cudaMemcpyDeviceToHost ) );
    #endif
    DumpData(DATA_CPU, Z_CPU,  "output.vtk", DATA_CPU.vel[current_field], DATA_CPU.pres[current_field], NULL);
    DumpProfiles(DATA_CPU, Z_CPU, DATA_CPU.av_u, visc, u_tau, "output.dat");

    if(intergal_time>0.){
        for(int iz=0; iz<NZ; iz++) intergal_av_u[iz] *= 1./intergal_time;
        DumpProfiles(DATA_CPU, Z_CPU, intergal_av_u.data(), visc, u_tau, "res.dat");
        double dudy = 1.5*(intergal_av_u[1].u[0]-0.)/(data_cpu::DZ[0])-0.5*(intergal_av_u[2].u[0]-intergal_av_u[1].u[0])/(data_cpu::DZ[1]);
        double u_fric = dudy<0. ? 0. : sqrt(visc*dudy);
        printf("u_fric: obtained=%.6f, expected=%.6f\n", u_fric, u_tau);
    }

    DATA_CPU.dealloc_all();
    DATA_CUDA.dealloc_all();
    return 0;
}
