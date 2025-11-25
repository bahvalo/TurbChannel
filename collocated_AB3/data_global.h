__DEVICE__ inline int XR(int i) { int j = i+1; if((j&(NX-1)) == 0) j-=NX; return j; }
__DEVICE__ inline int XL(int i) { int j = i; if((j&(NX-1)) == 0) j+=NX; return j-1; }
__DEVICE__ inline int YR(int i) { int j = i+NX; if((j&(NX*(NY-1)))==0) j-=NX*NY; return j; }
__DEVICE__ inline int YL(int i) { int j = i; if((j&(NX*(NY-1)))==0) j+=NX*NY; return j-NX; }
__DEVICE__ inline int ZR(int i) { return i+NX*NY; }
__DEVICE__ inline int ZL(int i) { return i-NX*NY; }
__DEVICE__ inline int Neighb(int i, int dir, int ilr){
    if(dir==0) return ilr ? XR(i) : XL(i);
    if(dir==1) return ilr ? YR(i) : YL(i);
    if(dir==2) return ilr ? ZR(i) : ZL(i);
    return -1; // error
}


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
    real *nu_t; // turbulent viscosity
    // Average by X and Y (as a function of Z)
    real9 *av_u;

    __DEVICE__ inline int IsWall(int i) const { return i<NX*NY || i>=N3D-NX*NY; }
    __DEVICE__ real GetCellHeight(int iz) const;
    __DEVICE__ real GetCellVolume(int in) const;

    // Enforce zero average pressure
    void NormalizePressure(real* f) const;

    // Integral values
    real CalcKineticEnergy(const real3* u) const;
    real3 CalcIntegral(const real3* u) const;
    // Average by X and Y
    void CalcVelocityProfile(const real3* u); // to av_u
    // CFL estimation. IsConvVisc=1 for convective only, =2 for viscous only, =3 for both
    __DEVICE__ real GetNuTAtNode(int i) const;
    __DEVICE__ real CalcCFL_loc(int i, const real3* u, double tau, int IsConvVisc) const;
    real CalcCFL(const real3* u, real tau, int IsConvVisc=3) const;

    // Main numerical methods
    void ApplyGradient(const real* a, real3* R);
    __DEVICE__ void ApplyGradient(int i, const real* a, real3* R);
    void UnapplyGradient(const real3* v, real* p);
    __DEVICE__ void ApplyDivergence(int i, const real3* v, real* dv);
    void ExplicitTerm(double t, const real3* u, const real* p, real3* kuhat) const;
    __DEVICE__ real3 CalcViscTermGalerkin(int i, const real3* v) const;
    void CalcFluxTerm_FD(double t, const real3* u, const real* p, int DoConv, int DoVisc, real3* kuhat) const;
    __DEVICE__ void CalcFluxTerm_FD(int i, const real3* u, int DoConv, int DoVisc, real3* kuhat) const;
    void ImplicitTerm(double t, const real3* u, const real* p, real3* ku);
    void ImplicitStage(double t, double tau_stage, const real3* ustar, const real* p, real3* u);

    // Turbulence
    void RecalcTurbVisc(const real3* u);
    __DEVICE__ void RecalcTurbVisc(int i, const real3* u);

    // Main function -- make one timestep
    void Step(double t_start, double tau, int current_field, int bdf_order);

    inline void alloc_all(){
        Vphys = alloc<fft_vphys_t>(N3D);
        #ifdef THIS_IS_CUDA
            // On CUDA, we use real-to-complex and complex-to-real FFT, so it takes less space
            Vspec = alloc<fft_vspec_t>((NX/2+1)*NY*NZ);
        #else
            Vspec = alloc<fft_vspec_t>(NX*NY*NZ);
        #endif
        for(int i=0; i<4; i++) vel[i] = alloc<real3>(N3D);
        for(int i=0; i<4; i++) flux[i] = alloc<real3>(N3D);
        for(int i=0; i<4; i++) pres[i] = alloc<real>(N3D);
        divv        = alloc<real>(N3D);
        if(EnableTurbVisc) nu_t = alloc<real>(N3D); else nu_t = NULL;
        cosphi[0]   = alloc<double>(NX);
        cosphi[1]   = alloc<double>(NY);
        av_u        = alloc<real9>(NZ);
    }
    inline void dealloc_all(){
        dealloc(Vphys);
        dealloc(Vspec);
        for(int i=0; i<4; i++) dealloc(vel[i]);
        for(int i=0; i<4; i++) dealloc(flux[i]);
        for(int i=0; i<4; i++) dealloc(pres[i]);
        dealloc(divv);
        if(nu_t) dealloc(nu_t);
        for(int idir=0; idir<2; idir++) dealloc(cosphi[idir]);
        dealloc(av_u);
    }
};

__CONSTANT__ double ZZ[NZ_MAX];
__CONSTANT__ double DZ[NZ_MAX];
__CONSTANT__ double MAT_A[NZ_MAX];
__CONSTANT__ struct data D;


