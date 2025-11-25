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
        using fft_vphys_t = cufftDoubleComplex;
        using fft_vspec_t = cufftDoubleComplex;
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

    // "Current" fields
    real3 *velocity;
    real *pressure, *qressure;
    // Fields at internal stages and other buffer arrays
    real3 *Ku[MAX_N_STAGES], *Kuhat[MAX_N_STAGES]; // velocity fluxes -- for each Runge-Kutta stage
    real *mujtilde[MAX_N_STAGES], *qj[MAX_N_STAGES]; // scalar data  -- for each Runge-Kutta stage
    real3 *nu1tilde; // initial momentum residual (for methods of type CK only)
    real *buf_scalar, *pj, *muj, *divv, *pjhat;
    real3 *buf_vector, *gradp, *u;
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

    // Main numerical methods
    void ApplyGradient(const real* a, real3* R);
    __DEVICE__ void ApplyGradient(int i, const real* a, real3* R);
    void UnapplyGradient(const real3* v, real* p);
    __DEVICE__ void ApplyDivergence(int i, const real3* v, real* dv);
    void ExplicitTerm(double t, const real3* u, const real* p, real3* kuhat) const;
    void CalcFluxTerm_FD(double t, const real3* u, const real* p, int DoConv, int DoVisc, real3* kuhat) const;
    __DEVICE__ void CalcFluxTerm_FD(int i, const real3* u, int DoConv, int DoVisc, real3* kuhat) const;
    void ImplicitTerm(double t, const real3* u, const real* p, real3* ku);
    void ImplicitStage(double t, double tau_stage, const real3* ustar, const real* p, real3* u);

    // Main function -- make one timestep. Input: solution at t_start, output: solution at t_start+tau (overwrites input)
    // For StabType=0, dpdt is ignored (may be not allocated)
    void Step(double t_start, double tau, real3* velocity, real* pressure, real* dpdt);

    inline void alloc_all(){
        Vphys = alloc<fft_vphys_t>(N3D);
        Vspec = alloc<fft_vspec_t>(N3D);
        for(int istage=0; istage<NStages; istage++) Ku[istage] = alloc<real3>(N3D);
        for(int istage=0; istage<NStages; istage++) Kuhat[istage] = alloc<real3>(N3D);
        for(int istage=0; istage<NStages; istage++) mujtilde[istage] = alloc<real>(N3D);
        for(int istage=0; istage<NStages; istage++) qj[istage] = alloc<real>(N3D);
        if(IMEXtype==tIMEXtype::IMEX_CK) nu1tilde = alloc<real3>(N3D);
        buf_scalar = alloc<real>(N3D);
        pj          = alloc<real>(N3D);
        muj         = alloc<real>(N3D);
        divv        = alloc<real>(N3D);
        pjhat       = alloc<real>(N3D);
        buf_vector  = alloc<real3>(N3D);
        gradp       = alloc<real3>(N3D);
        u           = alloc<real3>(N3D);
        cosphi[0]   = alloc<double>(NX);
        cosphi[1]   = alloc<double>(NY);
        velocity    = alloc<real3>(N3D);
        pressure    = alloc<real>(N3D);
        qressure    = alloc<real>(N3D);
        av_u        = alloc<real9>(NZ);
    }
    inline void dealloc_all(){
        dealloc(Vphys);
        dealloc(Vspec);
        for(int istage=0; istage<NStages; istage++) dealloc(Ku[istage]);
        for(int istage=0; istage<NStages; istage++) dealloc(Kuhat[istage]);
        for(int istage=0; istage<NStages; istage++) dealloc(mujtilde[istage]);
        for(int istage=0; istage<NStages; istage++) dealloc(qj[istage]);
        dealloc(nu1tilde);
        dealloc(buf_scalar);
        dealloc(pj);
        dealloc(muj);
        dealloc(divv);
        dealloc(pjhat);
        dealloc(buf_vector);
        dealloc(gradp);
        dealloc(u);
        for(int idir=0; idir<2; idir++) dealloc(cosphi[idir]);
        dealloc(velocity);
        dealloc(pressure);
        dealloc(qressure);
        dealloc(av_u);
    }
};

__CONSTANT__ double DZ[NZ_MAX];
__CONSTANT__ double MAT_A[NZ_MAX];
__CONSTANT__ struct data D;


