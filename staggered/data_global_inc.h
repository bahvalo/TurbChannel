// This header must be included withing a namespace (data_cpu or data_cuda)
// config.h, cufft.h, fftw3.h must be included outside the namespace

__DEVICE__ inline int XR(int i) { int j = i+1; if((j&(NX-1)) == 0) j-=NX; return j; }
__DEVICE__ inline int XL(int i) { int j = i; if((j&(NX-1)) == 0) j+=NX; return j-1; }
__DEVICE__ inline int YR(int i) { int j = i+NX; if((j&(NX*(NY-1)))==0) j-=NX*NY; return j; }
__DEVICE__ inline int YL(int i) { int j = i; if((j&(NX*(NY-1)))==0) j+=NX*NY; return j-NX; }
__DEVICE__ inline int ZR(int i) { return i+NX*NY; }
__DEVICE__ inline int ZL(int i) { return i-NX*NY; }

struct data : main_params{
    // FFTW data and wrappers
    #ifdef THIS_IS_CUDA
        cufftHandle p1, p2;
        #if USE_DOUBLE_PRECISION
            using fft_vphys_t = cufftDoubleReal;
            using fft_vspec_t = cufftDoubleComplex;
        #else
            using fft_vphys_t = cufftReal;
            using fft_vspec_t = cufftComplex;
        #endif
    #else
        fftw_plan p1, p2;
        using fft_vphys_t = fftw_complex;
        using fft_vspec_t = fftw_complex;
    #endif
    fft_vphys_t *Vphys;
    fft_vspec_t *Vspec;
    double* cosphi[2];
    __DEVICE__ void ThomasForPressure(int ixy);
    void InitFourier();
    void LinearSystemSolveFFT(real* x, const real* f);

    // Fields
    real3 *vel[4]; // layers: n-2, n-1, n, n+1
    real3 *flux[4]; // layers: n-2, n-1, n, buffer
    real *pres[4]; // layers: n-2, n-1, n, n+1
    real *divv;
    real *nu_t; // turbulent viscosity (may be not allocated)
    // Average by X and Y (as a function of Z) -- in the same precision as the main variables
    real9<real> *av_u;

    // Enforce zero average pressure
    void NormalizePressure(real* f) const;

    // Integral values
    real CalcKineticEnergy(const real3* u) const;
    real3 CalcIntegral(const real3* u) const;
    // Average by X and Y
    void CalcProfiles(const real3* u, const real* p); // to av_u
    // CFL estimation. IsConvVisc=1 for convective only, =2 for viscous only, =3 for both
    __DEVICE__ real CalcCFL_loc(int i, const real3* u, double tau, int IsConvVisc) const;
    real CalcCFL(const real3* u, real tau, int IsConvVisc=3) const;

    // Main numerical methods
    void ApplyGradient(const real* a, real3* R);
    __DEVICE__ void ApplyGradient(int i, const real* a, real3* R);
    void UnapplyGradient(const real3* v, real* p);
    __DEVICE__ void ApplyDivergence(int i, const real3* v, real* dv);

    __DEVICE__ real3 ConvFlux(int in, const real3* vel) const;
    template<tViscApproximation VA> __DEVICE__ real3 ViscFlux(int in, const real3* vel) const;
    void ExplicitTerm(const real3* u, real3* kuhat) const;
    void CalcFluxTerm_FD(const real3* u, int DoConv, int DoVisc, real3* kuhat) const;

    // Turbulence
    void RecalcTurbVisc(const real3* u);
    __DEVICE__ real CalcAbsS(int i, const real3* vel);
    __DEVICE__ void RecalcTurbVisc(int i, const real3* vel);

    // Main function -- make one timestep
    void Step(double t_start, double tau, int current_field, int bdf_order);

    void alloc_all();
    void dealloc_all();
};
