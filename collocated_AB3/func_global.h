// Implementation of methods, both executable on host and on device
// For CUDA, D is the data structure with pointers to the device memory
// For CPU, D is the data on host with pointers to the host memory

// Some functions are in separate files
#include "func_fluxes.h"

__DEVICE__ void data::ThomasForPressure(int ixy){
    #ifdef THIS_IS_CUDA
        // On CUDA, we use R2C/C2R or D2Z/Z2Z FFT, which results in a different stride
        #define STRIDE_X (NX/2+1)
        // cufftDoubleComplex is different to fftw_complex
        #define REAL(V) V.x
        #define IMAG(V) V.y
    #else
        #define STRIDE_X NX
        #define REAL(V) V[0]
        #define IMAG(V) V[1]
    #endif

    real MM;
    {
        int ix = ixy%STRIDE_X, iy = ixy/STRIDE_X;
        real cos_phi = cosphi[0][ix];
        real cos_psi = cosphi[1][iy];
        MM = (1-cos_phi)*hy/hx + (1-cos_psi)*hx/hy;
    }

    fft_vspec_t* x = Vspec + ixy;
    real p[NZ_MAX];

    {
        real b0 = -MAT_A[1];
        if(!FDmode) b0 += DZ[0]*MM;
        if(ixy==0) b0 *= 2; // pin
        real inv_b0 = 1/b0;
        REAL(x[0]) *= inv_b0;
        IMAG(x[0]) *= inv_b0;
        p[0] = MAT_A[1]*inv_b0;
    }

    const int stride = STRIDE_X*NY;
    for(int i=1; i<=NZ-1; i++){
        const real _MM = FDmode&&(i==0||i==NZ-1) ? real(0) : MM;
        real delta = 0; // b[i]
        if(i!=0) delta -= MAT_A[i] + DZ[i-1]*_MM;
        if(i!=NZ-1) delta -= MAT_A[i+1] + DZ[i]*_MM;
        delta -= MAT_A[i]*p[i-1];
        delta = 1/delta;
        if(i!=NZ-1) p[i] = MAT_A[i+1]*delta;
        REAL(x[i*stride]) = (REAL(x[i*stride])-MAT_A[i]*REAL(x[(i-1)*stride]))*delta;
        IMAG(x[i*stride]) = (IMAG(x[i*stride])-MAT_A[i]*IMAG(x[(i-1)*stride]))*delta;
    }
    for(int i=NZ-2; i>=0; i--){
        REAL(x[i*stride]) -= REAL(x[(i+1)*stride])*p[i];
        IMAG(x[i*stride]) -= IMAG(x[(i+1)*stride])*p[i];
    }
    #undef STRIDE_X
    #undef REAL
    #undef IMAG
}

#ifdef THIS_IS_CUDA
void data::InitFourier(){
    int n[2] = {NY,NX};
    const int stride = (NX/2+1)*NY;
    #if USE_DOUBLE_PRECISION
        CUFFT_CHECK_CRASH(cufftPlanMany(&p1, 2, n, NULL, 1, NX*NY, NULL, 1, stride, CUFFT_D2Z, NZ));
        CUFFT_CHECK_CRASH(cufftPlanMany(&p2, 2, n, NULL, 1, NX*NY, NULL, 1, stride, CUFFT_Z2D, NZ));
    #else
        CUFFT_CHECK_CRASH(cufftPlanMany(&p1, 2, n, NULL, 1, NX*NY, NULL, 1, stride, CUFFT_R2C, NZ));
        CUFFT_CHECK_CRASH(cufftPlanMany(&p2, 2, n, NULL, 1, NX*NY, NULL, 1, stride, CUFFT_C2R, NZ));
    #endif

    Forall1D([] __device__ (int i){
        if(i<NX) D.cosphi[0][i]=cos(2.*Pi*i/NX);
        if(i<NY) D.cosphi[1][i]=cos(2.*Pi*i/NY);
    }, NX+NY);
}

void data::LinearSystemSolveFFT(real* XX, const real* f){
    t_PressureSolver[0].beg();
    Forall1D([f]__device__(int in){ D.Vphys[in]=f[in]; });
    t_PressureSolver[1].beg();
    #if USE_DOUBLE_PRECISION
        CUFFT_CHECK_CRASH(cufftExecD2Z(p1, Vphys, Vspec));
    #else
        CUFFT_CHECK_CRASH(cufftExecR2C(p1, Vphys, Vspec));
    #endif
    #ifdef SYNC_AFTER_EACH_KERNEL
        GPU_CHECK_CRASH(cudaDeviceSynchronize());
    #endif
    t_PressureSolver[1].end();

    t_PressureSolver[2].beg();
    Forall1D([]__device__(int ixy){ D.ThomasForPressure(ixy); }, (NX/2+1)*NY);
    t_PressureSolver[2].end();
    t_PressureSolver[1].beg();
    #if USE_DOUBLE_PRECISION
        CUFFT_CHECK_CRASH(cufftExecZ2D(p2, Vspec, Vphys));
    #else
        CUFFT_CHECK_CRASH(cufftExecC2R(p2, Vspec, Vphys));
    #endif
    #ifdef SYNC_AFTER_EACH_KERNEL
        GPU_CHECK_CRASH(cudaDeviceSynchronize());
    #endif
    t_PressureSolver[1].end();

    real inv_NN = 1./(NX*NY);
    Forall1D([inv_NN, XX]__device__(int in){ XX[in]=D.Vphys[in]*inv_NN; }, N3D);
    #ifdef NORMALIZE_PRESSURE_AFTER_EACH_SOLVE
        NormalizePressure(XX);
    #endif
    t_PressureSolver[0].end();
}
#else
void data::InitFourier(){
    int n[2] = {NY,NX};
    // = fftw_plan_many_dft(Rank, n, howmany, A, inembed, istride, idist, A, onembed, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
    p1 = fftw_plan_many_dft(2, n, NZ, Vphys, NULL, 1, NX*NY, Vspec, NULL, 1, NX*NY, FFTW_FORWARD, FFTW_MEASURE);
    p2 = fftw_plan_many_dft(2, n, NZ, Vspec, NULL, 1, NX*NY, Vphys, NULL, 1, NX*NY, FFTW_BACKWARD, FFTW_MEASURE);

    for(int i=0; i<NX; i++) cosphi[0][i]=cos(2.*Pi*i/NX);
    for(int i=0; i<NY; i++) cosphi[1][i]=cos(2.*Pi*i/NY);
}

void data::LinearSystemSolveFFT(real* XX, const real* f){
    t_PressureSolver[0].beg();
    for(int in=0; in<N3D; in++) { Vphys[in][0]=f[in]; Vphys[in][1]=0.; }
    t_PressureSolver[1].beg();
    fftw_execute(p1);
    t_PressureSolver[1].end();

    t_PressureSolver[2].beg();
    Forall1D([](int ixy) { D.ThomasForPressure(ixy); }, NX*NY);
    t_PressureSolver[2].end();
    t_PressureSolver[1].beg();
    fftw_execute(p2);
    t_PressureSolver[1].end();

    double inv_NN = 1./(NX*NY);
    for(int in=0; in<N3D; in++) XX[in]=Vphys[in][0]*inv_NN;
    #ifdef NORMALIZE_PRESSURE_AFTER_EACH_SOLVE
        NormalizePressure(XX);
    #endif
    t_PressureSolver[0].end();
}
#endif

__DEVICE__ real data::GetCellHeight(int in) const{
    int iz = in/(NX*NY);
    if(iz==0) return real(0.5)*DZ[0];
    if(iz==NZ-1) return real(0.5)*DZ[NZ-2];
    return real(0.5)*(DZ[iz-1]+DZ[iz]);
}

__DEVICE__ real data::GetCellVolume(int in) const{
    return GetCellHeight(in)*hx*hy;
}


// Enforce the zero pressure average
void data::NormalizePressure(real* f) const{
    t_PressureSolver[3].beg();
    thrust::counting_iterator<int> I;
    real psum = thrust::transform_reduce(THRUST_POLICY, I, I+N3D, [f] __DEVICE__ (int i)->real { return f[i]*D.GetCellVolume(i); }, 0., thrust::plus<real>());
    real p_shift = psum*hx*hy / area;
    Forall1D([f,p_shift] __DEVICE__ (int i){ f[i] -= p_shift; });
    t_PressureSolver[3].end();
}

// Integral values
real data::CalcKineticEnergy(const real3* u) const{
    thrust::counting_iterator<int> I;
    return 0.5*thrust::transform_reduce(THRUST_POLICY, I, I+N3D, [u] __DEVICE__ (int i)->real { return DotProd(u[i],u[i])*D.GetCellVolume(i); }, real(0), thrust::plus<real>());
}

real3 data::CalcIntegral(const real3* u) const{
    thrust::counting_iterator<int> I;
    return thrust::transform_reduce(THRUST_POLICY, I, I+N3D, [u] __DEVICE__ (int i)->real3 { return u[i]*D.GetCellVolume(i); }, real3(), thrust::plus<real3>());
}

__DEVICE__ real data::GetNuTAtNode(int i) const{
    if(IsWall(i) || nu_t==NULL) return 0.0;
    return real(0.125)*(nu_t[i] + nu_t[XL(i)] + nu_t[YL(i)] + nu_t[XL(YL(i))] + nu_t[ZL(i)] + nu_t[XL(ZL(i))] + nu_t[YL(ZL(i))] + nu_t[XL(YL(ZL(i)))]);
}

__DEVICE__ real data::CalcCFL_loc(int i, const real3* u, double tau, int IsConvVisc) const{
    // Convection
    real inv_hz = 1/GetCellHeight(i);
    real CFL = 0;
    if(IsConvVisc&1){
        CFL += tau*(fabs(u[i][0])*inv_hx + fabs(u[i][1])*inv_hy + fabs(u[i][2])*inv_hz);
    }
    if(IsConvVisc&2){
        CFL += 4*(VISC_TERM_GALERKIN+1)*tau*(visc+GetNuTAtNode(i))*(inv_hx*inv_hx + inv_hy*inv_hy + inv_hz*inv_hz);
    }
    return CFL;
}

real data::CalcCFL(const real3* u, real tau, int IsConvVisc) const{
    thrust::counting_iterator<int> I;
    return thrust::transform_reduce(THRUST_POLICY, I, I+N3D, [u,tau,IsConvVisc] __DEVICE__ (int i)->real { return D.CalcCFL_loc(i, u, tau, IsConvVisc); },
           real(0), [] __DEVICE__ (real x, real y)->real { return x>y ? x : y; });
}

// Average by X and Y of a given vector field. Output: av_u
#ifdef THIS_IS_CUDA
#define VELPROF_BLK 64
__global__ void CalcVelocityProfile_kernel(int NZ, const real3* u, const real* nu_t){
    __shared__ float _shm[VELPROF_BLK*sizeof(real9)/4];
    real9* shm = (real9*)_shm;

    int iz = blockIdx.x;
    if(iz>=NZ) return; // should not happen

    real9 sum;
    if(threadIdx.x<VELPROF_BLK){
        for(int in=iz*NX*NY + threadIdx.x; in<(iz+1)*NX*NY; in+=VELPROF_BLK){
            sum.u += u[in];
            sum.uu[0] += u[in][0]*u[in][0];
            sum.uu[1] += u[in][1]*u[in][1];
            sum.uu[2] += u[in][2]*u[in][2];
            sum.uu[3] += u[in][0]*u[in][1];
            sum.uu[4] += u[in][0]*u[in][2];
            sum.uu[5] += u[in][1]*u[in][2];
            if(nu_t!=NULL) sum.nu_t += nu_t[in];
        }
        sum *= 1./(NX*NY);
        shm[threadIdx.x] = sum;
    }
    __syncthreads();

    #if VELPROF_BLK>=64
    if(threadIdx.x<32) shm[threadIdx.x] += shm[threadIdx.x+32];
    __syncthreads();
    #endif
    #if VELPROF_BLK>=32
    if(threadIdx.x<16) shm[threadIdx.x] += shm[threadIdx.x+16];
    __syncthreads();
    #endif
    #if VELPROF_BLK>=16
    if(threadIdx.x<8) shm[threadIdx.x] += shm[threadIdx.x+8];
    __syncthreads();
    #endif
    #if VELPROF_BLK>=8
    if(threadIdx.x<4) shm[threadIdx.x] += shm[threadIdx.x+4];
    __syncthreads();
    #endif
    #if VELPROF_BLK>=4
    if(threadIdx.x<2) shm[threadIdx.x] += shm[threadIdx.x+2];
    __syncthreads();
    #endif
    if(threadIdx.x==0){
        #if VELPROF_BLK>=2
        shm[0] += shm[1];
        #endif
        D.av_u[iz] = shm[0];
    }
}

void data::CalcVelocityProfile(const real3* u){
    const int BLK = VELPROF_BLK;
    const int GRID = NZ;
    CalcVelocityProfile_kernel<<<GRID,BLK>>>(NZ,u,nu_t);
    GPU_CHECK_CRASH(cudaGetLastError());
    #ifdef SYNC_AFTER_EACH_KERNEL
        GPU_CHECK_CRASH(cudaDeviceSynchronize());
    #endif
}
#else
void data::CalcVelocityProfile(const real3* u){
    Forall1D([u] __DEVICE__ (int iz){
        real9 sum;
        for(int in=iz*NX*NY; in<(iz+1)*NX*NY; in++){
            sum.u += u[in];
            sum.uu[0] += u[in][0]*u[in][0];
            sum.uu[1] += u[in][1]*u[in][1];
            sum.uu[2] += u[in][2]*u[in][2];
            sum.uu[3] += u[in][0]*u[in][1];
            sum.uu[4] += u[in][0]*u[in][2];
            sum.uu[5] += u[in][1]*u[in][2];
            if(D.nu_t!=NULL) sum.nu_t += D.nu_t[in];
        }
        sum *= real(1)/(NX*NY);
        D.av_u[iz] = sum;
    }, NZ);
}
#endif

// Gradient and L^{-1}D
void data::ApplyGradient(const real* a, real3* R){
    t_ApplyGradient.beg();
    Forall([a,R] __DEVICE__ (int i){ D.ApplyGradient(i,a,R); });
    t_ApplyGradient.end();
}

__DEVICE__ void data::ApplyGradient(int in, const real* a, real3* R){
    if(IsWall(in)){ R[in] = real3(); return; }
    R[in][0] = real(0.5)*inv_hx*(a[XR(in)]-a[XL(in)]);
    R[in][1] = real(0.5)*inv_hy*(a[YR(in)]-a[YL(in)]);
    R[in][2] = real(0.5)*(a[ZR(in)]-a[ZL(in)]) / GetCellHeight(in);
}

void data::UnapplyGradient(const real3* v, real* Lm1Dv){
    Forall([v] __DEVICE__ (int i){ D.ApplyDivergence(i,v,D.divv); });
    LinearSystemSolveFFT(Lm1Dv, divv);
}

__DEVICE__ void data::ApplyDivergence(int in, const real3* v, real* dv){
    real _dv = 0;
    real hz = GetCellHeight(in);
    _dv += real(0.5)*(v[XR(in)][0]-v[XL(in)][0])*hy*hz;
    _dv += real(0.5)*(v[YR(in)][1]-v[YL(in)][1])*hx*hz;
    if(in<NX*NY){
        _dv += real(0.5)*(v[ZR(in)][2]-v[in][2])*hx*hy;
    }
    else if(in>=(NZ-1)*NX*NY){
        _dv += real(0.5)*(v[in][2]-v[ZL(in)][2])*hx*hy;
    }
    else{
        _dv += real(0.5)*(v[ZR(in)][2]-v[ZL(in)][2])*hx*hy;
    }
    dv[in] = _dv;
}

void data::ImplicitTerm(double t, const real3* u, const real* p, real3* ku){
    Forall1D([ku] __DEVICE__ (int i){ ku[i]=real3(); });
}
void data::ImplicitStage(double t, double tau_stage, const real3* ustar, const real* p, real3* u){
    Forall1D([ustar,u] __DEVICE__ (int i){ u[i]=ustar[i]; });
}

// Turbulence
void data::RecalcTurbVisc(const real3* u){
    if(nu_t==NULL) return;
    Forall([u] __DEVICE__ (int i){ D.RecalcTurbVisc(i, u); });
}

__DEVICE__ inline real SQR(real x){ return x*x; }

__DEVICE__ void data::RecalcTurbVisc(int ie, const real3* u){
    int iz = ie/(NX*NY);
    if(iz>=NZ-1) return; // here ie is the element index, so the last plane is excessive

    const real hz = DZ[iz];

    real AbsS;
    {
        // corners of the element
        int in = ie;
        int in_x = XR(in);
        int in_y = YR(in);
        int in_xy = YR(in_x);
        int in_z = ZR(in);
        int in_xz = ZR(in_x);
        int in_yz = ZR(in_y);
        int in_xyz = ZR(in_xy);

        real3 gradu[3];
        gradu[0] = ((u[in_x]+u[in_xy]+u[in_xz]+u[in_xyz])-(u[in]+u[in_y]+u[in_z]+u[in_yz])) * (real(0.25)*inv_hx);
        gradu[1] = ((u[in_y]+u[in_xy]+u[in_yz]+u[in_xyz])-(u[in]+u[in_x]+u[in_z]+u[in_xz])) * (real(0.25)*inv_hy);
        gradu[2] = ((u[in_z]+u[in_xz]+u[in_yz]+u[in_xyz])-(u[in]+u[in_x]+u[in_y]+u[in_xy])) / (4*hz);

        gradu[0][1] = real(0.5)*(gradu[0][1]+gradu[1][0]);
        gradu[0][2] = real(0.5)*(gradu[0][2]+gradu[2][0]);
        gradu[1][2] = real(0.5)*(gradu[1][2]+gradu[2][1]);
        AbsS=SQR(gradu[0][0])+SQR(gradu[1][1])+SQR(gradu[2][2])+2*(SQR(gradu[0][1])+SQR(gradu[0][2])+SQR(gradu[1][2]));
        AbsS=sqrt(2*AbsS);
    }

    const real H = real(0.5)*(ZZ[NZ-1] - ZZ[0]);
    const real Z = real(0.5)*(ZZ[iz]+ZZ[iz+1]);
    const real dist_to_wall = (Z<H) ? Z : 2*H-Z;
    const real y_plus = dist_to_wall*inv_length_scale;

    const real C_SMAG = 0.1;
    const real C_exp = 0.04*0.04*0.04;
    const real f_VD = 1-exp(-C_exp*y_plus*y_plus*y_plus); // van Driest multiplier
    const real V = hx*hy*hz;
    real Delta_LES = C_SMAG*pow(V,C1_3);
    if(Delta_LES>real(0.41)*dist_to_wall) Delta_LES=real(0.41)*dist_to_wall;
    nu_t[ie] = f_VD*Delta_LES*Delta_LES*AbsS;
}


void data::Step(double t_start, double tau, int current_field, int bdf_order){
    if(bdf_order<1 || bdf_order>3) { printf("Wrong bdf_order\n"); exit(0); }
    int next_field = (current_field+1)%4;
    const real inv_tau = real(1./tau);
    real bd0=(bdf_order==3) ? real(1.5+C1_3) : (bdf_order==2) ? real(1.5) : real(1.0);
    const real tau_over_bd0 = tau / bd0;

    if(EnableTurbVisc) RecalcTurbVisc(vel[current_field]);

    // Explicit fluxes at t_n
    ExplicitTerm(t_start, vel[current_field], pres[current_field], flux[current_field]);

    // Extrapolated explicit fluxes + BDF terms. Use flux[next_field] as buffer
    if(bdf_order==1){
        Forall1D([inv_tau,current_field] __DEVICE__ (int in){
            int next_field = (current_field+1)%4;
            D.flux[next_field][in] = D.flux[current_field][in] + inv_tau*D.vel[current_field][in];
        });
    }
    if(bdf_order==2){
        Forall1D([inv_tau,current_field] __DEVICE__ (int in){
            int next_field = (current_field+1)%4;
            int prev_field = (current_field+3)%4;
            D.flux[next_field][in] = 2.*D.flux[current_field][in] - D.flux[prev_field][in]
                 + inv_tau*(2*D.vel[current_field][in] - real(0.5)*D.vel[prev_field][in]);
        });
    }
    if(bdf_order==3){
        Forall1D([inv_tau,current_field] __DEVICE__ (int in){
            int next_field = (current_field+1)%4;
            int prev2_field = (current_field+2)%4;
            int prev_field = (current_field+3)%4;
            D.flux[next_field][in] = 3*D.flux[current_field][in] - 3*D.flux[prev_field][in] +  D.flux[prev2_field][in]
                 + inv_tau*(3*D.vel[current_field][in] - real(1.5)*D.vel[prev_field][in] + C1_3*D.vel[prev2_field][in]);
        });
    }

    if(StabType==0){
        // Pressure solver
        UnapplyGradient(flux[next_field], pres[next_field]);

        // Velocity correction
        ApplyGradient(pres[next_field], vel[next_field]);
        Forall1D([tau_over_bd0,next_field] __DEVICE__ (int in){
            D.vel[next_field][in] = (D.flux[next_field][in] - D.vel[next_field][in]) * tau_over_bd0;
        });
    }
    else{
        // Use vel[next_field] as a buffer
        ApplyGradient(pres[current_field], vel[next_field]);
        Forall1D([next_field] __DEVICE__ (int in) { D.flux[next_field][in] -= D.vel[next_field][in]; });

        // Pressure solver for the pressure correction
        UnapplyGradient(flux[next_field], pres[next_field]);

        ApplyGradient(pres[next_field], vel[next_field]); // gradient of the pressure increment
        Forall1D([current_field,next_field] __DEVICE__ (int in) { D.pres[next_field][in] += D.pres[current_field][in]; });

        // Velocity correction
        Forall1D([tau_over_bd0,next_field] __DEVICE__ (int in){
            D.vel[next_field][in] = (D.flux[next_field][in] - D.vel[next_field][in]) * tau_over_bd0;
        });
    }
}

