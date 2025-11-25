// Implementation of methods, both executable on host and on device
// For CUDA, D is the data structure with pointers to the device memory
// For CPU, D is the data on host with pointers to the host memory

__DEVICE__ void data::ThomasForPressure(int ixy){
    real MM;
    {
        int ix = ixy%NX, iy = ixy/NX;
        real cos_phi = cosphi[0][ix];
        real cos_psi = cosphi[1][iy];
        MM = (1.-cos_phi)*hy/hx + (1.-cos_psi)*hx/hy;
    }

    fft_vspec_t* x = Vspec + ixy;
    real p[NZ_MAX];

    {
        real b0 = -MAT_A[1];
        if(!FDmode) b0 += DZ[0]*MM;
        if(ixy==0) b0 *= 2.; // pin
        real inv_b0 = 1./b0;
        REAL(x[0]) *= inv_b0;
        IMAG(x[0]) *= inv_b0;
        p[0] = MAT_A[1]*inv_b0;
    }

    for(int i=1; i<=NZ-1; i++){
        const real _MM = FDmode&&(i==0||i==NZ-1) ? 0. : MM;
        real delta = 0.; // b[i]
        if(i!=0) delta -= MAT_A[i] + DZ[i-1]*_MM;
        if(i!=NZ-1) delta -= MAT_A[i+1] + DZ[i]*_MM;
        delta -= MAT_A[i]*p[i-1];
        delta = 1./delta;
        if(i!=NZ-1) p[i] = MAT_A[i+1]*delta;
        REAL(x[i*NX*NY]) = (REAL(x[i*NX*NY])-MAT_A[i]*REAL(x[(i-1)*NX*NY]))*delta;
        IMAG(x[i*NX*NY]) = (IMAG(x[i*NX*NY])-MAT_A[i]*IMAG(x[(i-1)*NX*NY]))*delta;
    }
    for(int i=NZ-2; i>=0; i--){
        REAL(x[i*NX*NY]) -= REAL(x[(i+1)*NX*NY])*p[i];
        IMAG(x[i*NX*NY]) -= IMAG(x[(i+1)*NX*NY])*p[i];
    }
}

#ifdef THIS_IS_CUDA
void data::InitFourier(){
    int n[2] = {NY,NX};
    CUFFT_CHECK_CRASH(cufftPlanMany(&p1, 2, n, NULL, 1, NX*NY, NULL, 1, NX*NY, CUFFT_Z2Z, NZ));
    CUFFT_CHECK_CRASH(cufftPlanMany(&p2, 2, n, NULL, 1, NX*NY, NULL, 1, NX*NY, CUFFT_Z2Z, NZ));

    Forall1D([] __device__ (int i){
        if(i<NX) D.cosphi[0][i]=cos(2.*Pi*i/NX);
        if(i<NY) D.cosphi[1][i]=cos(2.*Pi*i/NY);
    }, NX+NY);
}

void data::LinearSystemSolveFFT(real* XX, const real* f){
    t_PressureSolver[0].beg();
    Forall1D([f]__device__(int in){ REAL(D.Vphys[in])=f[in]; IMAG(D.Vphys[in])=0.; });
    t_PressureSolver[1].beg();
    CUFFT_CHECK_CRASH(cufftExecZ2Z(p1, Vphys, Vspec, CUFFT_FORWARD));
    #ifdef SYNC_AFTER_EACH_KERNEL
    GPU_CHECK_CRASH(cudaDeviceSynchronize());
    #endif
    t_PressureSolver[1].end();

    t_PressureSolver[2].beg();
    Forall1D([]__device__(int ixy){ D.ThomasForPressure(ixy); }, NX*NY);
    t_PressureSolver[2].end();
    t_PressureSolver[1].beg();
    CUFFT_CHECK_CRASH(cufftExecZ2Z(p2, Vspec, Vphys, CUFFT_INVERSE));
    #ifdef SYNC_AFTER_EACH_KERNEL
    GPU_CHECK_CRASH(cudaDeviceSynchronize());
    #endif
    t_PressureSolver[1].end();

    real inv_NN = 1./(NX*NY);
    Forall1D([inv_NN, XX]__device__(int in){ XX[in]=REAL(D.Vphys[in])*inv_NN; }, N3D);
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
    for(int ixy=0; ixy<NX*NY; ixy++) ThomasForPressure(ixy);
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
    if(iz==0) return 0.5*DZ[0];
    if(iz==NZ-1) return 0.5*DZ[NZ-2];
    return 0.5*(DZ[iz-1]+DZ[iz]);
}

__DEVICE__ real data::GetCellVolume(int in) const{
    return GetCellHeight(in)*hx*hy;
}


// Enforce the zero pressure average
void data::NormalizePressure(real* f) const{
    t_PressureSolver[3].beg();
    thrust::counting_iterator<int> I;
    real psum = thrust::transform_reduce(THRUST_POLICY, I, I+N3D, [f] __DEVICE__ (int i)->real { return f[i]*D.GetCellVolume(i); }, 0., thrust::plus<real>());
    real p_shift = psum / area;
    Forall1D([f,p_shift] __DEVICE__ (int i){ f[i] -= p_shift; });
    t_PressureSolver[3].end();
}

// Integral values
real data::CalcKineticEnergy(const real3* u) const{
    thrust::counting_iterator<int> I;
    return 0.5*thrust::transform_reduce(THRUST_POLICY, I, I+N3D, [u] __DEVICE__ (int i)->real { return DotProd(u[i],u[i])*D.GetCellVolume(i); }, 0., thrust::plus<real>());
}

real3 data::CalcIntegral(const real3* u) const{
    thrust::counting_iterator<int> I;
    return thrust::transform_reduce(THRUST_POLICY, I, I+N3D, [u] __DEVICE__ (int i)->real3 { return u[i]*D.GetCellVolume(i); }, real3(), thrust::plus<real3>());
}

// Average by X and Y of a given vector field. Output: av_u
#ifdef THIS_IS_CUDA
#define VELPROF_BLK 64
__global__ void CalcVelocityProfile_kernel(int NZ, const real3* u){
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
            //sum.nu_t += nu_t[in]*m;
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
    CalcVelocityProfile_kernel<<<GRID,BLK>>>(NZ,u);
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
            //sum.nu_t += nu_t[in]*m;
        }
        sum *= 1./(NX*NY);
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
    R[in][0] = 0.5*inv_hx*(a[XR(in)]-a[XL(in)]);
    R[in][1] = 0.5*inv_hy*(a[YR(in)]-a[YL(in)]);
    R[in][2] = 0.5*(a[ZR(in)]-a[ZL(in)]) / GetCellHeight(in);
}

void data::UnapplyGradient(const real3* v, real* Lm1Dv){
    Forall([v] __DEVICE__ (int i){ D.ApplyDivergence(i,v,D.divv); });
    LinearSystemSolveFFT(Lm1Dv, divv);
}

__DEVICE__ void data::ApplyDivergence(int in, const real3* v, real* dv){
    real _dv = 0.;
    real hz = GetCellHeight(in);
    _dv += 0.5*(v[XR(in)][0]-v[XL(in)][0])*hy*hz;
    _dv += 0.5*(v[YR(in)][1]-v[YL(in)][1])*hx*hz;
    if(in<NX*NY){
        _dv += 0.5*(v[ZR(in)][2]-v[in][2])*hx*hy;
    }
    else if(in>=(NZ-1)*NX*NY){
        _dv += 0.5*(v[in][2]-v[ZL(in)][2])*hx*hy;
    }
    else{
        _dv += 0.5*(v[ZR(in)][2]-v[ZL(in)][2])*hx*hy;
    }
    dv[in] = _dv;
}

void data::ExplicitTerm(double t, const real3* u, const real* p, real3* kuhat) const{
    t_ExplicitTerm.beg();
    Forall([t,u,p,kuhat] __DEVICE__ (int i){ D.CalcFluxTerm_FD(i,u,true,true,kuhat); });
    t_ExplicitTerm.end();
}
void data::CalcFluxTerm_FD(double t, const real3* u, const real* p, int DoConv, int DoVisc, real3* kuhat) const{
    Forall([u,DoConv,DoVisc,kuhat] __DEVICE__ (int i){ D.CalcFluxTerm_FD(i,u,DoConv,DoVisc,kuhat); });
}

__DEVICE__ void data::CalcFluxTerm_FD(int in, const real3* u, int DoConv, int DoVisc, real3* kuhat) const{
    real3 sumflux;
    if(!IsWall(in)){
        int iz = in/(NX*NY);
        double inv_hz = 2./(DZ[iz-1]+DZ[iz]);
        if(DoConv){ // Convection
            const real3 inv_MeshStep(inv_hx, inv_hy, inv_hz);

            for(int idir=0; idir<3; idir++){
                const double invh = inv_MeshStep[idir];
                int jn = Neighb(in, idir, 0);
                int kn = Neighb(in, idir, 1);

                real3 conv_div = 0.5*(u[kn]*u[kn][idir] - u[jn]*u[jn][idir]);
                real3 conv_adv = u[in][idir] * 0.5*(u[kn]-u[jn]);
                sumflux -= 0.5*(conv_div + conv_adv) * invh;
            }
        }

        // Viscosity
        if(DoVisc){
            sumflux += visc*inv_hx*inv_hx*(u[XR(in)]+u[XL(in)]-2.*u[in]);
            sumflux += visc*inv_hy*inv_hy*(u[YR(in)]+u[YL(in)]-2.*u[in]);
            sumflux += visc*((u[ZR(in)]-u[in])/DZ[iz] - (u[in]-u[ZL(in)])/DZ[iz-1])*inv_hz;
        }
        // Source term
        sumflux[0] += sour_dpdx;
    }
    kuhat[in] = sumflux;
}

void data::ImplicitTerm(double t, const real3* u, const real* p, real3* ku){
    Forall1D([ku] __DEVICE__ (int i){ ku[i]=real3(); });
}
void data::ImplicitStage(double t, double tau_stage, const real3* ustar, const real* p, real3* u){
    Forall1D([ustar,u] __DEVICE__ (int i){ u[i]=ustar[i]; });
}

void data::Step(double t_start, double tau, real3* velocity, real* pressure, real* dpdt){
    const double ass = ButcherI[NStages*NStages-1]; // a_{ss}
    const double inv_ass = 1./ass;
    const double tau_stage = tau*ass;
    const double inv_tau_stage = 1./tau_stage;
    const double alpha = alpha_tau/tau;

    if(IMEXtype==tIMEXtype::IMEX_CK || IMEXtype==tIMEXtype::IMEX_ARS){
        ExplicitTerm(t_start, velocity, pressure, Kuhat[0]);
        ApplyGradient(pressure, gradp);
        Forall1D([] __DEVICE__ (int in){ D.Ku[0][in] = real3(); D.Kuhat[0][in] -= D.gradp[in]; });
    }
    if(IMEXtype==tIMEXtype::IMEX_CK){
        ImplicitTerm(t_start, velocity, pressure, Ku[0]);
        Forall1D([alpha] __DEVICE__ (int in){ D.nu1tilde[in]=D.Ku[0][in]+D.Kuhat[0][in]+alpha*D.velocity[in]; });
    }
    const int j0 = IMEXtype==tIMEXtype::IMEX_A ? 0 : 1;
    for(int istage=j0; istage<NStages; istage++){
        double c_i = 0.0;
        for(int jstage=0; jstage<=istage; jstage++) c_i += ButcherI[istage*NStages+jstage];
        const double time_stage = t_start + c_i*tau;

        // Velocity step
        Forall1D([istage,tau] __DEVICE__ (int in){
            D.buf_vector[in] = D.velocity[in]; // here buf_vector = u_{j,*}
            for(int jstage=0; jstage<istage; jstage++){
                const double qi = tau*D.ButcherI[istage*D.NStages+jstage];
                const double qe = tau*D.ButcherE[istage*D.NStages+jstage];
                D.buf_vector[in] += qi*D.Ku[jstage][in] + qe*D.Kuhat[jstage][in];
            }
        });

        ImplicitStage(time_stage, tau_stage, buf_vector, pressure, u);
        Forall1D([istage,inv_tau_stage] __DEVICE__ (int in){ D.Ku[istage][in] = (D.u[in] - D.buf_vector[in]) * inv_tau_stage; });

        // Explicit velocity term. Pressure gradient will be substracted later
        ExplicitTerm(time_stage, u, pressure, Kuhat[istage]);

        // Pressure step.
        // Evaluating the right-hand side of the pressure system in buf_vector
        if(StabType==0){
            Forall1D([istage,j0,inv_ass,alpha,tau_stage,pressure] __DEVICE__ (int in){
                D.pjhat[in] = 0.;
                D.muj[in] = pressure[in];
                if(D.IMEXtype==tIMEXtype::IMEX_CK) D.muj[in] *= (1. - D.alpha_tau*D.ButcherI[istage*D.NStages]);
                for(int jstage=j0; jstage<istage; jstage++) D.muj[in] += inv_ass*D.ButcherI[istage*D.NStages+jstage]*D.mujtilde[jstage][in];
                D.buf_scalar[in] = D.muj[in] - alpha*tau_stage*pressure[in];
            });
        }
        else{
            Forall1D([j0,istage,tau,alpha,tau_stage,pressure,dpdt,inv_ass] __DEVICE__ (int in){
                D.muj[in] = tau_stage*dpdt[in];
                if(D.IMEXtype==tIMEXtype::IMEX_CK) D.muj[in] *= (1. - D.alpha_tau*D.ButcherI[istage*D.NStages]);
                D.pjhat[in] = pressure[in];
                if(D.IMEXtype==tIMEXtype::IMEX_CK) D.pjhat[in] += tau*D.ButcherI[istage*D.NStages]*dpdt[in];
                for(int jstage=j0; jstage<istage; jstage++){
                    D.muj[in] += inv_ass*D.ButcherI[istage*D.NStages+jstage]*D.mujtilde[jstage][in];
                    D.pjhat[in] += tau*D.ButcherI[istage*D.NStages+jstage]*D.qj[jstage][in];
                }
                D.buf_scalar[in] = D.muj[in] + D.pjhat[in] - alpha*tau_stage*tau_stage*dpdt[in];
            });
        }
        ApplyGradient(buf_scalar, buf_vector);

        Forall1D([alpha,istage] __DEVICE__ (int in){
            D.buf_vector[in] = D.Ku[istage][in] + D.Kuhat[istage][in] - D.buf_vector[in] + alpha*D.velocity[in];
            if(D.IMEXtype==tIMEXtype::IMEX_CK) D.buf_vector[in] -= D.d[istage]*D.nu1tilde[in];
        });

        // Here we finally solve the pressure system
        UnapplyGradient(buf_vector, mujtilde[istage]);

        Forall1D([istage,alpha,tau_stage,inv_tau_stage,pressure,dpdt] __DEVICE__ (int in){
            D.pj[in] = D.mujtilde[istage][in] + D.buf_scalar[in];
            D.mujtilde[istage][in] -= alpha*tau_stage*(D.StabType ? tau_stage*dpdt[in] : pressure[in]);
            if(D.StabType) D.qj[istage][in] = inv_tau_stage*(D.pj[in]-D.pjhat[in]);
        });
        ApplyGradient(pj, buf_vector);
        Forall1D([istage] __DEVICE__ (int in){ D.Kuhat[istage][in] -= D.buf_vector[in]; });
    }

    // Final solution
    Forall1D([tau,velocity,pressure,dpdt] __DEVICE__ (int in){
        for(int jstage=0; jstage<D.NStages; jstage++){
            const double qi = tau*D.ButcherI[D.NStages*D.NStages+jstage];
            const double qe = tau*D.ButcherE[D.NStages*D.NStages+jstage];
            velocity[in] += qi*D.Ku[jstage][in] + qe*D.Kuhat[jstage][in];
        }
        pressure[in] = D.pj[in]; // take the last obtained pressure
        if(D.StabType) dpdt[in] = D.qj[D.NStages-1][in];
    });
}
