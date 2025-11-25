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
#include <thrust/iterator/counting_iterator.h>

// For debugging or timing
#define SYNC_AFTER_EACH_KERNEL
#define NORMALIZE_PRESSURE_AFTER_EACH_SOLVE

// Floating-point type (float or double)
#define real double
#include "real3.h"
#include "imex.h" // IMEX Runge-Kutta methods
#include "mytimer.h"

// Mesh dimensions in X and Y. Must be powers of 2
#define NX 128
#define NY 64
#define NZ_MAX 256 // maximal number of mesh nodes in Z (for static arrays). Should be a multiple of 16 for the sake of alignment

// Kernel launch configuration (block size)
// For 1D block grid
#define BLK_1D 256
// For 3D block grid. BLK_X and BLK_Y must be dividers of NX and NY, respectively
#define BLK_X 8
#define BLK_Y 8
#define BLK_Z 4

// Floating-point constants
#define C2_3    0.66666666666666666666
#define C4_3    1.33333333333333333333
#define C1_12   0.08333333333333333333
#define C1_60   0.01666666666666666666
#define C1_90   0.01111111111111111111
#define C49_18  2.72222222222222222222
#define Pi      3.14159265358979323846

// Parameters of governing equations, discretizations, etc.
// Must be plain data, no nontrivial constructors allowed
struct main_params{
    // Parameters of governing equations
    double visc; // constant viscosity coefficient
    double sour_dpdx; // source term in the X-component of the momentum equation

    // Mesh parameters
    double xmin, ymin; // offsets, which do not matter
    double hx, hy, inv_hx, inv_hy, area;
    int NZ, N3D;

    // Time discretization parameters
    tIMEXtype IMEXtype;
    int NStages; // number of stages of the RK method
    double ButcherE[BUTCHER_TABLE_MAX_SIZE];
    double ButcherI[BUTCHER_TABLE_MAX_SIZE]; // Butcher tables for the explicit and implicit methods (of size (NStages+1)*NStages)
    double d[MAX_N_STAGES];
    double alpha_tau_max[2]; // Stability limit in alpha_tau for StabType=0 and StabType=1
    double alpha_tau; // alpha = alpha_tau / tau. A recommended value is set automatically
    int StabType; // (0) Du+Sp=0 or (1) Du+Sdp/dt=0

    // Spatial discretization parameters
    int FDmode;
};

// Structure for averaging
struct real9{
    real3 u;
    real uu[6]; // XX, YY, ZZ, XY, XZ, YZ
    real nu_t;
    __device__ __host__ inline real9(){ uu[0]=uu[1]=uu[2]=uu[3]=uu[4]=uu[5]=nu_t=0.; }
    __device__ __host__ inline real9& operator*=(const real &x){ u*=x; for(int i=0; i<6; i++) uu[i]*=x; nu_t*=x; return *this; }
    __device__ __host__ inline real9& operator+=(const real9& o){ u+=o.u; for(int i=0; i<6; i++) uu[i]+=o.uu[i]; nu_t+=o.nu_t; return *this; }
};
__device__ __host__ inline real9 operator*(const real &a, const real9 &b){real9 R(b); return R *= a;}

// Timers used in methods of the data structure
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
#define REAL(V) V[0]
#define IMAG(V) V[1]
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
#undef REAL
#undef IMAG
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
#define REAL(V) V.x
#define IMAG(V) V.y
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
#undef REAL
#undef IMAG
#undef THRUST_POLICY
#undef __DEVICE__
#undef __CONSTANT__
#undef THIS_IS_CUDA


// ---------------------------------------------------------------------------
// CPU-only code
// ---------------------------------------------------------------------------

static double Z_CPU[NZ_MAX];
#define DATA_CPU data_cpu::D

void DumpData(const char* fname, const real3* velocity, const real* pressure, const real* qressure){
    const int N[3] = {NX, NY, DATA_CPU.NZ}, NN = DATA_CPU.N3D;
    FILE* f = fopen(fname, "wt");
    fprintf(f, "# vtk DataFile Version 2.0\n");
    fprintf(f, "Volume example\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET RECTILINEAR_GRID\n");
    fprintf(f, "DIMENSIONS %i %i %i\n", N[0], N[1], N[2]);
    for(int idir=0; idir<3; idir++){
        fprintf(f, "%c_COORDINATES %i float\n", 'X'+idir, N[idir]);
        for(int i=0; i<N[idir]; i++) fprintf(f, "%e ", (idir==0) ? DATA_CPU.xmin+DATA_CPU.hx*i : (idir==1) ? DATA_CPU.ymin+DATA_CPU.hy*i : Z_CPU[i]);
        fprintf(f, "\n");
    }
    fprintf(f, "POINT_DATA %i\n", NN);
    fprintf(f, "SCALARS volume_scalars float %i\n", 5);
    fprintf(f, "LOOKUP_TABLE default\n");
    for(int in=0; in<NN; ++in){
        fprintf(f, "%e %e %e %e %e", velocity[in][0], velocity[in][1], velocity[in][2], pressure[in], qressure[in]);
        fprintf(f, "\n");
    }
    fclose(f);
}


static void DumpProfiles(const real9* av, double visc, double u_fric, const char* fname, int IsZplus=0){
    int NZ = DATA_CPU.NZ;
    const double* Z = Z_CPU;
    FILE* F = fopen(fname, "wt");

    for(int iz=0; iz<NZ; iz++){
        double H = 0.5*(Z[NZ-1] - Z[0]);
        double dist_to_wall = (Z[iz]<H) ? Z[iz] : 2.*H-Z[iz];
        double y_plus = dist_to_wall*u_fric;
        if(!IsZplus) y_plus /= visc;

        real3 u = av[iz].u / u_fric;
        double uu[6];
        for(int i=0; i<6; i++) uu[i] = av[iz].uu[i] / (u_fric*u_fric);
        uu[0] = std::max(0., uu[0]-u[0]*u[0]);
        uu[1] = std::max(0., uu[1]-u[1]*u[1]);
        uu[2] = std::max(0., uu[2]-u[2]*u[2]);
        uu[3] -= u[0]*u[1];
        uu[4] -= u[0]*u[2];
        uu[5] -= u[1]*u[2];

        double nu_t; // turbulent viscosity is defined at half-integer points, need to average
        if(iz==0) nu_t = av[0].nu_t;
        else if(iz==NZ-1) nu_t = av[iz-1].nu_t;
        else nu_t = 0.5*(av[iz].nu_t + av[iz-1].nu_t);

        fprintf(F, "%e %e %e %e %e %e %e\n", y_plus, u[0], sqrt(uu[0]), sqrt(uu[1]), sqrt(uu[2]), uu[4], nu_t/visc);
    }
    fclose(F);
}

int IsPowerOfTwo(int i){
    if(i<=0) return 0;
    while(i>1){ if(i&1) return 0; i>>=1; };
    return 1;
}

int main( void ){
    // Checks
    if(!IsPowerOfTwo(NX) || !IsPowerOfTwo(NY) || NX%BLK_X || NY%BLK_Y) { printf("WRONG DEFINES\n"); exit(0); }
    printf("sizeof(data_cuda::data) = %lu (single pointer is %lu)\n", sizeof(data_cuda::data), sizeof(real*));
    // Problem parameters
    const double Re_tau = 180.; // u_tau*H/visc
    const double H = 1.; // channel half-height
    const double visc = 1./180.; // kinematic viscosity (affects the velocity scale only)
    const double u_tau = Re_tau*visc; // expected friction velocity
    const double sour_dpdx = u_tau*u_tau / H; // source in the momentum equation per unit volume
    const double Hx = 4.*Pi*H, Hy = 4.*Pi*H/3, Hz = 2.*H; // domain size
    const double xmin = 0., ymin = -0.5*Hy, zmin = 0.; // offset, does not matter

    std::vector<double> Z_plus; // mesh coordinates in normal direction ("y+")
    {
        #define Z Z_plus
        const double y1 = 1.; // first step
        const double ybulk = 10.; // max step
        const double coeff = 1.3; // geometric progression coefficient

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
    const int NZ = Z_plus.size();
    if(NZ > NZ_MAX) { printf("WRONG NZ_MAX=%i (actual number of nodes = %i)\n", NZ_MAX, NZ); exit(0); }
    for(int i=0; i<NZ; i++) Z_CPU[i] = Z_plus[i]*visc;
    for(int i=0; i<NZ-1; i++) data_cpu::DZ[i] = Z_CPU[i+1]-Z_CPU[i];
    const int N3D = NX*NY*NZ;

    {
        main_params& MD = DATA_CPU;
        // Copying Butcher tables and related data to DATA_CPU and DATA_CUDA
        ARS_343 IMEX;
        int Butcher_Table_Size = IMEX.NStages*(IMEX.NStages+1);
        MD.IMEXtype = IMEX.IMEXtype;
        MD.NStages = IMEX.NStages;
        for(int i=0; i<Butcher_Table_Size; i++) MD.ButcherE[i] = IMEX.ButcherE[i];
        for(int i=0; i<Butcher_Table_Size; i++) MD.ButcherI[i] = IMEX.ButcherI[i];
        std::vector<double> d = tIMEXMethod::CalcD(IMEX.NStages, IMEX.ButcherI);
        for(int i=0; i<IMEX.NStages; i++) MD.d[i] = d[i];
        for(int i=0; i<2; i++) MD.alpha_tau_max[i] = IMEX.alpha_tau_max[i];
        MD.StabType = 1;
        MD.alpha_tau = 0.5*IMEX.alpha_tau_max[MD.StabType];

        // Other parameters
        MD.visc = visc;
        MD.sour_dpdx = sour_dpdx;

        // Mesh parameters
        MD.NZ = NZ;
        MD.N3D = NX*NY*NZ;
        MD.hx = Hx/NX;
        MD.hy = Hy/NY;
        MD.inv_hx = 1./MD.hx;
        MD.inv_hy = 1./MD.hy;
        MD.area = Hx*Hy*(Z_CPU[NZ-1]-Z_CPU[0]);
        MD.xmin = xmin;
        MD.ymin = ymin;

        // Spatial discretization
        MD.FDmode = 1;

        // Copy the main parameters to DATA_CPU and DATA_CUDA
        //*(static_cast<main_params*>(&DATA_CPU)) = MD;
    }

    // Data on host with device pointers
    struct data_cuda::data DATA_CUDA;
    // Copy the main parameters to DATA_CUDA
    *(static_cast<main_params*>(&DATA_CUDA)) = *(static_cast<main_params*>(&DATA_CPU));

    // Alloc global memory
    DATA_CPU.alloc_all();
    DATA_CPU.InitFourier();

    DATA_CUDA.alloc_all(); // alloc memory on device

    // Send the structure to DATA_ON_DEVICE
    GPU_CHECK_CRASH( cudaMemcpyToSymbol(data_cuda::D, &DATA_CUDA, sizeof(data_cuda::data)) );
    DATA_CUDA.InitFourier();
    GPU_CHECK_CRASH( cudaMemcpyToSymbol(data_cuda::D, &DATA_CUDA, sizeof(data_cuda::data)) );

    GPU_CHECK_CRASH( cudaMemcpyToSymbol(data_cuda::DZ, data_cpu::DZ, sizeof(double)*NZ) );

    {
        double* A = data_cpu::MAT_A;
        A[0]=0.;
        for(int j=1; j<NZ; j++) A[j]=DATA_CPU.hx*DATA_CPU.hy/data_cpu::DZ[j-1];
        if(DATA_CPU.FDmode){ A[1]*=0.5; A[NZ-1]*=0.5; }
    }
    GPU_CHECK_CRASH( cudaMemcpyToSymbol(data_cuda::MAT_A, data_cpu::MAT_A, sizeof(double)*NZ) );

    // Comparison of specific subroutines on CPU and GPU
    #if 1
    {
        for(int in=0; in<N3D; in++) DATA_CPU.pressure[in] = sin(double(in)*in);
        GPU_CHECK_CRASH( cudaMemcpy( DATA_CUDA.pressure, DATA_CPU.pressure, N3D*sizeof(real), cudaMemcpyHostToDevice ) );
        DATA_CUDA.ApplyGradient(DATA_CUDA.pressure, DATA_CUDA.velocity);
        DATA_CPU.ApplyGradient(DATA_CPU.pressure, DATA_CPU.velocity);
        GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.buf_vector, DATA_CUDA.velocity, N3D*sizeof(real3), cudaMemcpyDeviceToHost ) );
        cudaDeviceSynchronize();
        double err = 0.;
        for(int in=0; in<N3D; in++) err += abs(DATA_CPU.buf_vector[in]-DATA_CPU.velocity[in]);
        printf("Error in ApplyGradient: %f\n", err);
    }
    {
        for(int in=0; in<N3D; ++in){
            int i[3] = {in%NX, (in/NX)%NY, in/(NX*NY)};
            double x = i[0]*DATA_CPU.hx, /*y = i[1]*DATA_CPU.hy,*/ z = Z_plus[i[2]]*visc;
            double zplus = std::min(Z_plus[i[2]], Re_tau*2-Z_plus[i[2]]);
            DATA_CPU.velocity[in][0] = 0.5*(z-zmin)*(zmin+Hz-z)*sour_dpdx/visc;
            DATA_CPU.velocity[in][1] = 1e-1*sin(0.5*x)*zplus*exp(-0.01*zplus*zplus);
            DATA_CPU.velocity[in][2] = 0.;
            for(int idir=0; idir<3; idir++) if(i[2]!=0 && i[2]!=NZ-1) DATA_CPU.velocity[in][idir] += 1e-1*sin(double(in)*in);
        }
        GPU_CHECK_CRASH( cudaMemcpy( DATA_CUDA.velocity, DATA_CPU.velocity, N3D*sizeof(real3), cudaMemcpyHostToDevice ) );
        DATA_CUDA.UnapplyGradient(DATA_CUDA.velocity, DATA_CUDA.pressure);
        DATA_CPU.UnapplyGradient(DATA_CPU.velocity, DATA_CPU.pressure);
        GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.buf_scalar, DATA_CUDA.pressure, N3D*sizeof(real), cudaMemcpyDeviceToHost ) );
        cudaDeviceSynchronize();
        double err = 0.;
        for(int in=0; in<N3D; in++) err += fabs(DATA_CPU.pressure[in]-DATA_CPU.buf_scalar[in]);
        printf("Error in UnapplyGradient: %f\n", err);
    }
    #endif

    // Get the number of timesteps
    double TimeMax = 0.10000000001; // maximal integration time
    double tau = 0.8e-3;
    int MaxTimeSteps = 1000000;

    // Initial data
    for(int in=0; in<N3D; ++in){
        int i[3] = {in%NX, (in/NX)%NY, in/(NX*NY)};
        double x = i[0]*DATA_CPU.hx, /*y = i[1]*DATA_CPU.hy,*/ z = Z_plus[i[2]]*visc;
        double zplus = std::min(Z_plus[i[2]], Re_tau*2-Z_plus[i[2]]);
        DATA_CPU.velocity[in][0] = 0.5*(z-zmin)*(zmin+Hz-z)*sour_dpdx/visc;
        DATA_CPU.velocity[in][1] = 1e-1*sin(0.5*x)*zplus*exp(-0.01*zplus*zplus);
        DATA_CPU.velocity[in][2] = 0.;
        for(int idir=0; idir<3; idir++) if(i[2]!=0 && i[2]!=NZ-1) DATA_CPU.velocity[in][idir] += 1e-1*double(rand())/RAND_MAX;
        DATA_CPU.pressure[in] = 0.;
        DATA_CPU.qressure[in] = 0.;
    }

    //#define WORK_ON_GPU // uncomment for CUDA solver

    #ifdef WORK_ON_GPU
        #define FlowSolver DATA_CUDA
    #else
        #define FlowSolver DATA_CPU
        omp_set_num_threads(2);
    #endif

    #ifdef WORK_ON_GPU
    GPU_CHECK_CRASH( cudaMemcpy( DATA_CUDA.velocity, DATA_CPU.velocity, N3D*sizeof(real3), cudaMemcpyHostToDevice ) );
    GPU_CHECK_CRASH( cudaMemcpy( DATA_CUDA.pressure, DATA_CPU.pressure, N3D*sizeof(real), cudaMemcpyHostToDevice ) );
    GPU_CHECK_CRASH( cudaMemcpy( DATA_CUDA.qressure, DATA_CPU.qressure, N3D*sizeof(real), cudaMemcpyHostToDevice ) );
    cudaDeviceSynchronize(); // for the correct timing
    #endif

    // Time integration
    printf("Start of time integration (N=%i,%i,%i)\n", NX, NY, NZ);
    FILE* FILE_LOG = fopen("log.dat", "wt");
    int iTimeStep=0, isFinalStep=0;
    real3 FlowRate_prev = FlowSolver.CalcIntegral(FlowSolver.velocity)/DATA_CPU.area;
    std::vector<real9> intergal_av_u(NZ);

    MyTimer t_Total, t_Step, t_velprof, t_reduction, t_velprof_copy, t_output;
    t_Total.beg();
    double t = 0.;
    for(iTimeStep=0; iTimeStep < MaxTimeSteps && !isFinalStep; iTimeStep++){
        t_Step.beg();
        FlowSolver.Step(iTimeStep*tau, tau, FlowSolver.velocity, FlowSolver.pressure, FlowSolver.qressure);
        t_Step.end();

        t_reduction.beg();
        real3 FlowRate = FlowSolver.CalcIntegral(FlowSolver.velocity)/DATA_CPU.area;
        t_reduction.end();
        t_velprof.beg();
        FlowSolver.CalcVelocityProfile(FlowSolver.velocity);
        t_velprof.end();
        #ifdef WORK_ON_GPU
            t_velprof_copy.beg();
            GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.av_u, DATA_CUDA.av_u, NZ*sizeof(real9), cudaMemcpyDeviceToHost ) );
            t_velprof_copy.end();
        #endif
        for(int iz=0; iz<NZ; iz++) intergal_av_u[iz] += tau*DATA_CPU.av_u[iz];

        t_output.beg();
        t = (iTimeStep+1)*tau;
        if(t>=TimeMax) isFinalStep=1;
        double dudy_current = (DATA_CPU.av_u[1].u[0]-0.)/data_cpu::DZ[0];
        double u_fric_current = dudy_current<0. ? 0. : sqrt(dudy_current*visc);
        double dudy_integral = intergal_av_u[1].u[0]/(data_cpu::DZ[0]*t);
        double u_fric_integral = dudy_integral<0. ? 0. : sqrt(dudy_integral*visc);

        int DoPrintStdout = (iTimeStep%10==0) || isFinalStep;
        int DoPrintFile   = (iTimeStep%10==0) || isFinalStep;
        if(DoPrintStdout) printf("T=%f Ubulk=%.04f Ufric=%.04f UfricAv=%.04f\n", t, FlowRate[0], u_fric_current, u_fric_integral);
        if(DoPrintFile) fprintf(FILE_LOG, "%f %f %f %f\n", t, FlowRate[0], u_fric_current, u_fric_integral);

        int DoWriteOutput1 = (iTimeStep!=0 && iTimeStep%10000==0) || isFinalStep;
        int DoWriteOutput2 = DoWriteOutput1;//(iTimeStep!=0 && iTimeStep%200==0) || isFinalStep;
        if(DoWriteOutput1){
            char fname[256];
            sprintf(fname, "q%05i.vtk", iTimeStep);
            #ifdef WORK_ON_GPU
            GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.velocity, DATA_CUDA.velocity, N3D*sizeof(real3), cudaMemcpyDeviceToHost ) );
            GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.pressure, DATA_CUDA.pressure, N3D*sizeof(real), cudaMemcpyDeviceToHost ) );
            GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.qressure, DATA_CUDA.qressure, N3D*sizeof(real), cudaMemcpyDeviceToHost ) );
            #endif
            DumpData(fname, DATA_CPU.velocity, DATA_CPU.pressure, DATA_CPU.qressure);
        }
        if(DoWriteOutput2){
            char fname[256];
            sprintf(fname, "r%05i.dat", iTimeStep);
            DumpProfiles(DATA_CPU.av_u, visc, u_tau, fname);
        }
        t_output.end();
        FlowRate_prev = FlowRate;
    }
    fclose(FILE_LOG);

    cudaDeviceSynchronize(); // for the correct timing
    t_Total.end();
    printf("Time: total %.04f, step %.04f, expl %.04f, grad %.04f, pres %.04f (F=%.04f T=%.04f N=%.04f), velprof %.04f (copy %.04f), reduction %.04f, output %.04f\n", t_Total.timer, t_Step.timer, t_ExplicitTerm.timer, t_ApplyGradient.timer, t_PressureSolver[0].timer, t_PressureSolver[1].timer, t_PressureSolver[2].timer, t_PressureSolver[3].timer, t_velprof.timer, t_velprof_copy.timer, t_reduction.timer, t_output.timer);

    #ifdef WORK_ON_GPU
    GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.velocity, DATA_CUDA.velocity, N3D*sizeof(real3), cudaMemcpyDeviceToHost ) );
    GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.pressure, DATA_CUDA.pressure, N3D*sizeof(real), cudaMemcpyDeviceToHost ) );
    GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.qressure, DATA_CUDA.qressure, N3D*sizeof(real), cudaMemcpyDeviceToHost ) );
    #endif

    DumpData("output.vtk", DATA_CPU.velocity, DATA_CPU.pressure, DATA_CPU.qressure);

    if(t>0.){
        for(int iz=0; iz<NZ; iz++) intergal_av_u[iz] *= 1./t;
        DumpProfiles(intergal_av_u.data(), visc, u_tau, "res.dat");
        double dudy = 1.5*(intergal_av_u[1].u[0]-0.)/(data_cpu::DZ[0])-0.5*(intergal_av_u[2].u[0]-intergal_av_u[1].u[0])/(data_cpu::DZ[1]);
        double u_fric = dudy<0. ? 0. : sqrt(visc*dudy);
        printf("u_fric: obtained=%.6f, expected=%.6f\n", u_fric, u_tau);
    }

    DATA_CPU.dealloc_all();
    DATA_CUDA.dealloc_all();
    return 0;
}
