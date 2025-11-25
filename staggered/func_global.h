// ---------------------------------------------------------------------------
// Warning! This header should be included only in func_*.cpp, inside a namespace
// ---------------------------------------------------------------------------
// Implementation of methods, both executable on host and on device
// For CUDA, D is the data structure with pointers to the device memory
// For CPU, D is the data on host with pointers to the host memory
// ---------------------------------------------------------------------------

// Floating-point constants. Define only once, even if this header is included twice in one file
#ifndef CONSTANTS_DEFINED
#define CONSTANTS_DEFINED
#define C1_90   real(0.01111111111111111111)
#define C1_60   real(0.01666666666666666666)
#define C1_36   real(0.02777777777777777777)
#define C1_27   real(0.03703703703703703703)
#define C1_18   real(0.05555555555555555555)
#define C1_12   real(0.08333333333333333333)
#define C1_9    real(0.11111111111111111111)
#define C1_8    real(0.125                 )
#define C1_6    real(0.16666666666666666666)
#define C1_3    real(0.33333333333333333333)
#define C2_3    real(0.66666666666666666666)
#define C9_8    real(1.125                 )
#define C4_3    real(1.33333333333333333333)
#define C49_18  real(2.72222222222222222222)
#define Pi      real(3.14159265358979323846)
#endif

__DEVICE__ inline int XRR(int i) { return XR(XR(i)); }
__DEVICE__ inline int XLL(int i) { return XL(XL(i)); }
__DEVICE__ inline int YRR(int i) { return YR(YR(i)); }
__DEVICE__ inline int YLL(int i) { return YL(YL(i)); }
__DEVICE__ inline int ZRR(int i) { return ZR(ZR(i)); }
__DEVICE__ inline int ZLL(int i) { return ZL(ZL(i)); }

__DEVICE__ inline int XRRR(int i) { return XR(XR(XR(i))); }
__DEVICE__ inline int XLLL(int i) { return XL(XL(XL(i))); }
__DEVICE__ inline int YRRR(int i) { return YR(YR(YR(i))); }
__DEVICE__ inline int YLLL(int i) { return YL(YL(YL(i))); }
__DEVICE__ inline int ZRRR(int i) { return ZR(ZR(ZR(i))); }
__DEVICE__ inline int ZLLL(int i) { return ZL(ZL(ZL(i))); }

__DEVICE__ inline int Neighb(int i, int dir, int ilr){
    if(dir==0) return ilr ? XR(i) : XL(i);
    if(dir==1) return ilr ? YR(i) : YL(i);
    if(dir==2) return ilr ? ZR(i) : ZL(i);
    return -1; // error
}

__DEVICE__ inline real SQR(real x){ return x*x; }
__DEVICE__ inline int IMAX0(int x){ return x>0 ? x : 0; }


// ---------------------------------------------------------------------------
// Gradient/divergence approximations, pressure solver
// ---------------------------------------------------------------------------

#ifdef THIS_IS_CUDA
    // cufftDoubleComplex is different to fftw_complex
    #define REAL(V) V.x
    #define IMAG(V) V.y
    // On CUDA, we use R2C/C2R or D2Z/Z2Z FFT, which results in a different stride
    #define STRIDE_X (NX/2+1)
#else
    #define REAL(V) V[0]
    #define IMAG(V) V[1]
    #define STRIDE_X NX
#endif

#if(CONV_APPR_TYPE==CONV_HW)
__DEVICE__ void data::ThomasForPressure(int ixy){
    real MM;
    {
        int ix = ixy%STRIDE_X, iy = ixy/STRIDE_X;
        real cos_phi = cosphi[0][ix];
        real cos_psi = cosphi[1][iy];
        MM = 2*((1-cos_phi)*inv_hx*inv_hx + (1-cos_psi)*inv_hy*inv_hy);
    }

    fft_vspec_t* x = Vspec + ixy;
    real p[NZ_MAX];

    {
        real b0 = - DZ[0]*MM - MAT_A[1];
        if(ixy==0) b0 *= 2; // pin
        real inv_b0 = 1/b0;
        REAL(x[0]) *= inv_b0;
        IMAG(x[0]) *= inv_b0;
        p[0] = MAT_A[1]*inv_b0;
    }

    const int stride = STRIDE_X*NY;
    for(int i=1; i<=NZ-1; i++){
        real delta = - MAT_A[i] - DZ[i]*MM;
        if(i!=NZ-1) delta -= MAT_A[i+1]; // now delta is b[i]
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
}
#endif

#if(CONV_APPR_TYPE==CONV_VV)
__DEVICE__ void data::ThomasForPressure(int ixy){
    const int stride = STRIDE_X*NY;
    fft_vspec_t* x = Vspec+ixy;

    const int N = NZ, w = 3;
    const real tiny = 1e-30;
    real buf[NZ_MAX*w];

    // Forward substitution
    for(int i=0; i<N; i++){
        // Form the row `i` of the matrix
        real cur_line[2*w+1] = {};
        real sum = 0;
        // below diagonal
        for(int k=0; k<w; k++){
            if(i+k-w<0 || i+k-w>=N) continue;
            cur_line[k] = MAT_A[i*w+k];
            sum += MAT_A[i*w+k];
        }
        // elements above diagonal - by symmetry
        for(int k=1; k<=w; k++){
            if(i+k<0 || i+k>=N) continue;
            cur_line[w+k] = MAT_A[(i+k)*w + w-k];
            sum += MAT_A[(i+k)*w + w-k];
        }
        // diagonal elements
        if(i<=2) cur_line[w] = MAT_D[i];
        else if(i>=N-3) cur_line[w] = MAT_D[i-(N-3)+3];
        else cur_line[w] = - sum; // using zero row sum
        if(ixy==0 && i==0) cur_line[w] *= 2; // pin
        //if(ixy==0) printf("diag[%i]=%f\n", i, cur_line[w]);

        // Add the diagonal element
        {
            int ix = ixy%STRIDE_X, iy = ixy/STRIDE_X;
            real sin_phi_2 = sin(Pi*ix/NX); // sin(phi/2)
            real sin_3phi_2 = 3*sin_phi_2 - 4*sin_phi_2*sin_phi_2*sin_phi_2;
            real ci = DZ[i]*sin_phi_2 - C1_27*DZW[i]*sin_3phi_2;
            cur_line[w] -= 4*inv_hx*inv_hx * ci*ci / (DZ[i] - C1_9*DZW[i]);

            sin_phi_2 = sin(Pi*iy/NY); // sin(phi/2)
            sin_3phi_2 = 3*sin_phi_2 - 4*sin_phi_2*sin_phi_2*sin_phi_2;
            ci = DZ[i]*sin_phi_2 - C1_27*DZW[i]*sin_3phi_2;
            cur_line[w] -= 4*inv_hy*inv_hy * ci*ci / (DZ[i] - C1_9*DZW[i]);
        }

        // Substracting the previous lines
        for(int j=-w; j<0; j++){ // j<0 is the offset to the previous line
            if(i+j<0) continue; // no such line
            real mult = -cur_line[w+j];
            // formally, we should also set cur_line[w+j]=0, but this is pointless
            for(int k=1; k<=w; k++){
                if(i+j+k>=N) continue; // no such column
                cur_line[w+j+k] += mult*buf[(i+j)*w + k-1]; // line(i) += mult*line(i+j)
            }
            REAL(x[i*stride]) += mult*REAL(x[(i+j)*stride]);
            IMAG(x[i*stride]) += mult*IMAG(x[(i+j)*stride]);
        }
        // Normalizing the line
        if(fabs(cur_line[w]) < tiny) { REAL(x[0])=-1e50; return; } // Division by zero
        {
            real mult = real(1)/cur_line[w];
            for(int k=1; k<=w; k++){
                if(i+k>=N) continue; // no such column
                buf[i*w + k-1] = cur_line[w+k] * mult;
            }
            REAL(x[i*stride]) *= mult;
            IMAG(x[i*stride]) *= mult;
        }
    }
    // Backward substitution
    for(int i=N-1; i>=0; i--){
        for(int j=1; j<=w; j++){ // j>0 is the offset to the line below
            if(i+j>=N) continue;
            REAL(x[i*stride]) -= buf[i*w+j-1]*REAL(x[(i+j)*stride]);
            IMAG(x[i*stride]) -= buf[i*w+j-1]*IMAG(x[(i+j)*stride]);
            // i-th line of the matrix is replaced by the i-th line of the unit matrix. No need to write this to M
        }
    }
}
#endif

#undef REAL
#undef IMAG
#undef STRIDE_X

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
    Forall1D([](int ixy) { D.ThomasForPressure(ixy); }, NXY);
    t_PressureSolver[2].end();
    t_PressureSolver[1].beg();
    fftw_execute(p2);
    t_PressureSolver[1].end();

    double inv_NN = 1./NXY;
    for(int in=0; in<N3D; in++) XX[in]=Vphys[in][0]*inv_NN;
    #ifdef NORMALIZE_PRESSURE_AFTER_EACH_SOLVE
        NormalizePressure(XX);
    #endif
    t_PressureSolver[0].end();
}
#endif

// Gradient
void data::ApplyGradient(const real* a, real3* ga){
    t_ApplyGradient.beg();
    Forall([a,ga] __DEVICE__ (int i){ D.ApplyGradient(i,a,ga); });
    t_ApplyGradient.end();
}

void data::UnapplyGradient(const real3* v, real* Lm1Dv){
    Forall([v] __DEVICE__ (int i){ D.ApplyDivergence(i,v,D.divv); });
    LinearSystemSolveFFT(Lm1Dv, divv);
}


#if(CONV_APPR_TYPE==CONV_HW)
__DEVICE__ void data::ApplyGradient(int in, const real* a, real3* ga){
    int iz = in/NXY;
    ga[in][0] = inv_hx*(a[in]-a[XL(in)]);
    ga[in][1] = inv_hy*(a[in]-a[YL(in)]);
    if(iz==0) ga[in][2] = 0; // boundary condition at z=zmin
    else ga[in][2] = (a[in]-a[ZL(in)]) * (2/(DZ[iz]+DZ[iz-1]));
}

// div(v)*dz at cell `in`
__DEVICE__ void data::ApplyDivergence(int in, const real3* v, real* dv){
    real hz = DZ[in/NXY];
    dv[in] =
        (v[XR(in)][0]-v[in][0])*inv_hx*hz +
        (v[YR(in)][1]-v[in][1])*inv_hy*hz +
        ((in<(NZ-1)*NX*NY) ? (v[ZR(in)][2]-v[in][2]) : (-v[in][2])); // w=0 on z=zmax
}
#endif

#if(CONV_APPR_TYPE==CONV_VV)
__DEVICE__ void data::ApplyGradient(int in, const real* a, real3* ga){
    int iz = in/NXY;
    ga[in][0] = inv_hx*(GC1[iz]*(a[in]-a[XL(in)]) + GC2[iz]*(a[XR(in)]-a[XL(XL(in))]));
    ga[in][1] = inv_hy*(GC1[iz]*(a[in]-a[YL(in)]) + GC2[iz]*(a[YR(in)]-a[YL(YL(in))]));
    if(iz==0) ga[in][2] = 0; // gradient is not defined at z=zmin
    else ga[in][2] = ((a[in]-a[ZL(in)]) - C1_27*((iz==NZ-1 ? -a[in] : a[ZR(in)])-(iz==1 ? -a[ZL(in)] : a[ZL(ZL(in))]))) * GCZ[iz];
}

// div(v)*(dz-dzw/9) at cell `in`
__DEVICE__ void data::ApplyDivergence(int in, const real3* v, real* dv){
    const int iz = in/NXY;
    real dudx = inv_hx*(DZ[iz]*(v[XR(in)][0]-v[in][0]) - C1_27*DZW[iz]*(v[XR(XR(in))][0]-v[XL(in)][0]));
    real dvdy = inv_hy*(DZ[iz]*(v[YR(in)][1]-v[in][1]) - C1_27*DZW[iz]*(v[YR(YR(in))][1]-v[YL(in)][1]));
    real wr = (iz==NZ-1) ? real(0) : v[ZR(in)][2];
    real wrr = (iz==NZ-1) ? v[in][2] : (iz==NZ-2) ? real(0) : v[ZR(ZR(in))][2]; // zero mass flux on boundary, symmetry reflection for the -1 node
    real wll = (iz==0) ? v[ZR(in)][2] : (iz==1) ? real(0) : v[ZL(in)][2];
    dv[in] = dudx + dvdy + (wr-v[in][2]) - C1_27*(wrr-wll);
}
#endif

// ---------------------------------------------------------------------------
// Convective and diffusive terms
// ---------------------------------------------------------------------------

#define u(X) vel[(X)][0]
#define v(X) vel[(X)][1]
#define w(X) vel[(X)][2]

#if(CONV_APPR_TYPE==CONV_HW)
__DEVICE__ real3 data::ConvFlux(int in, const real3* vel) const{
    real3 sumflux;
    int iz = in/NXY;
    real inv_hz = 1/DZ[iz];
    real inv_hz_alt = real(2)/(DZ[iz]+(iz==0 ? real(0) : DZ[iz-1]));

    sumflux[0] -= ( SQR(u(in) + u(XR(in))) - SQR(u(in) + u(XL(in))) )*inv_hx;
    sumflux[0] -= ( (u(in)+u(YR(in)))*(v(YR(in))+v(YR(XL(in)))) - (u(in)+u(YL(in)))*(v(in)+v(XL(in))) )*inv_hy;
    if(iz!=NZ-1) sumflux[0] -= (u(in)+u(ZR(in)))*(w(ZR(in))+w(ZR(XL(in))))*inv_hz;
    if(iz!=0) sumflux[0] += (u(in)+u(ZL(in)))*(w(in)+w(XL(in)))*inv_hz;

    sumflux[1] -= ( (v(in)+v(XR(in)))*(u(XR(in))+u(XR(YL(in)))) - (v(in)+v(XL(in)))*(u(in)+u(YL(in))) )*inv_hx;
    sumflux[1] -= (SQR(v(in) + v(YR(in))) - SQR(v(in) + v(YL(in))))*inv_hy;
    if(iz!=NZ-1) sumflux[1] -= (v(in)+v(ZR(in)))*(w(ZR(in))+w(ZR(YL(in))))*inv_hz;
    if(iz!=0) sumflux[1] += (v(in)+v(ZL(in)))*(w(in)+w(YL(in)))*inv_hz;

    //if(iz==0) sumflux[2] = 0;
    if(iz!=0){ // for z=zmin, the fluxes in z are zero
        sumflux[2] -= ((w(in)+w(XR(in)))*(u(XR(in))*DZ[iz]+u(XR(ZL(in)))*DZ[iz-1]) - (w(in)+w(XL(in)))*(u(in)*DZ[iz]+u(ZL(in))*DZ[iz-1]))*inv_hx*inv_hz_alt;
        sumflux[2] -= ((w(in)+w(YR(in)))*(v(YR(in))*DZ[iz]+v(YR(ZL(in)))*DZ[iz-1]) - (w(in)+w(YL(in)))*(v(in)*DZ[iz]+v(ZL(in))*DZ[iz-1]))*inv_hy*inv_hz_alt;
        sumflux[2] -= SQR(w(in)+(iz==NZ-1?real(0):w(ZR(in))))*(inv_hz_alt); // w=0 on z=zmax assumed
        sumflux[2] += SQR(w(in)+w(ZL(in)))*(inv_hz_alt); // w=0 is kept on z=zmin, so no extra condition required
    }
    sumflux *= real(0.25); // due to the averaging
    return sumflux;
}
#endif

#if(CONV_APPR_TYPE==CONV_VV)
__DEVICE__ real3 data::ConvFlux(int in, const real3* vel) const{
    real3 sumflux;
    int iz = in/NXY;

    { // all 'bar' variables are multiplied by 16, and 'barbar' variables are multiplied by 16/9
        real ubar_xr = DZ[iz]*hy*( 9*(u(in)+u(XR(in))) - (u(XL(in))+u(XRR(in))) );
        real ubar_xl = DZ[iz]*hy*( 9*(u(XL(in))+u(in)) - (u(XLL(in))+u(XR(in))) );
        real ubarbar_xr = DZW[iz]*hy*( 9*(u(XR(in))+u(XRR(in))) - (u(in)+u(XRRR(in))) );
        real ubarbar_xl = DZW[iz]*hy*( 9*(u(XLL(in))+u(XL(in))) - (u(XLLL(in))+u(in)) );
        sumflux[0] += ubar_xr*(u(in)+u(XR(in))) - C1_27*ubarbar_xr*(u(in)+u(XRRR(in))) -
                      ubar_xl*(u(XL(in))+u(in)) + C1_27*ubarbar_xl*(u(XLLL(in))+u(in));
    } // compared to [V&V, 2003], this expression is multiplied by 32/(3^5)
    {
        real vbar_yr = DZ[iz]*hx*( 9*(v(YR(in))+v(XL(YR(in)))) - (v(XLL(YR(in)))+v(XR(YR(in)))) );
        real vbar_yl = DZ[iz]*hx*( 9*(v(  (in))+v(XL(  (in)))) - (v(XLL(  (in)))+v(XR(  (in)))) );
        real vbarbar_yr = DZW[iz]*hx*( 9*(v(YRR(in))+v(XL(YRR(in)))) - (v(XLL(YRR(in)))+v(XR(YRR(in)))) );
        real vbarbar_yl = DZW[iz]*hx*( 9*(v( YL(in))+v(XL( YL(in)))) - (v(XLL( YL(in)))+v(XR( YL(in)))) );
        sumflux[0] += vbar_yr*(u(in)+u(YR(in))) - C1_27*vbarbar_yr*(u(in)+u(YRRR(in))) -
                      vbar_yl*(u(YL(in))+u(in)) + C1_27*vbarbar_yl*(u(YLLL(in))+u(in));
    }
    {
        real wbar_zr, wbar_zl, wbarbar_zr, wbarbar_zl;
        if(iz==0) wbar_zl = 0;
        else wbar_zl = hx*hy*( 9*(w(  (in))+w(XL(  (in)))) - (w(XLL(  (in)))+w(XR(  (in)))) );
        if(iz==NZ-1) wbar_zr = 0;
        else wbar_zr = hx*hy*( 9*(w(ZR(in))+w(XL(ZR(in)))) - (w(XLL(ZR(in)))+w(XR(ZR(in)))) );
        if(iz==1) wbarbar_zl = 0;
        else if(iz==0) wbarbar_zl = wbar_zr;
        else wbarbar_zl = hx*hy*( 9*(w( ZL(in))+w(XL( ZL(in)))) - (w(XLL( ZL(in)))+w(XR( ZL(in)))) );
        if(iz==NZ-2) wbarbar_zr = 0;
        else if(iz==NZ-1) wbarbar_zr = wbar_zl;
        else wbarbar_zr = hx*hy*( 9*(w(ZRR(in))+w(XL(ZRR(in)))) - (w(XLL(ZRR(in)))+w(XR(ZRR(in)))) );
        real u_zr = iz==NZ-1 ? real(0) : u(in)+u(ZR(in));
        real u_zrr = u(in) + (iz==NZ-3 ? -u(ZRR(in)) : iz==NZ-2 ? real(0) : iz==NZ-1 ? u(ZLL(in)) : u(ZRRR(in)));
        real u_zl = iz==0 ? real(0) : u(ZL(in))+u(in);
        real u_zll = u(in) + (iz==2 ? -u(ZLL(in)) : iz==1 ? real(0) : iz==0 ? u(ZRR(in)) : u(ZLLL(in)));
        sumflux[0] += wbar_zr*u_zr - C1_27*wbarbar_zr*u_zrr -
                      wbar_zl*u_zl + C1_27*wbarbar_zl*u_zll;
    }

    {
        real ubar_xr = DZ[iz]*hy*( 9*(u(XR(in))+u(YL(XR(in)))) - (u(YLL(XR(in)))+u(YR(XR(in)))) );
        real ubar_xl = DZ[iz]*hy*( 9*(u(  (in))+u(YL(  (in)))) - (u(YLL(  (in)))+u(YR(  (in)))) );
        real ubarbar_xr = DZW[iz]*hy*( 9*(u(XRR(in))+u(YL(XRR(in)))) - (u(YLL(XRR(in)))+u(YR(XRR(in)))) );
        real ubarbar_xl = DZW[iz]*hy*( 9*(u( XL(in))+u(YL( XL(in)))) - (u(YLL( XL(in)))+u(YR( XL(in)))) );
        sumflux[1] += ubar_xr*(v(in)+v(XR(in))) - C1_27*ubarbar_xr*(v(in)+v(XRRR(in))) -
                      ubar_xl*(v(XL(in))+v(in)) + C1_27*ubarbar_xl*(v(XLLL(in))+v(in));
    }
    {
        real vbar_yr = DZ[iz]*hx*( 9*(v(in)+v(YR(in))) - (v(YL(in))+v(YRR(in))) );
        real vbar_yl = DZ[iz]*hx*( 9*(v(YL(in))+v(in)) - (v(YLL(in))+v(YR(in))) );
        real vbarbar_yr = DZW[iz]*hx*( 9*(v(YR(in))+v(YRR(in))) - (v(in)+v(YRRR(in))) );
        real vbarbar_yl = DZW[iz]*hx*( 9*(v(YLL(in))+v(YL(in))) - (v(YLLL(in))+v(in)) );
        sumflux[1] += vbar_yr*(v(in)+v(YR(in))) - C1_27*vbarbar_yr*(v(in)+v(YRRR(in))) -
                      vbar_yl*(v(YL(in))+v(in)) + C1_27*vbarbar_yl*(v(YLLL(in))+v(in));
    }
    {
        real wbar_zr, wbar_zl, wbarbar_zr, wbarbar_zl;
        if(iz==0) wbar_zl = 0;
        else wbar_zl = hx*hy*( 9*(w(  (in))+w(YL(  (in)))) - (w(YLL(  (in)))+w(YR(  (in)))) );
        if(iz==NZ-1) wbar_zr = 0;
        else wbar_zr = hx*hy*( 9*(w(ZR(in))+w(YL(ZR(in)))) - (w(YLL(ZR(in)))+w(YR(ZR(in)))) );
        if(iz==1) wbarbar_zl = 0;
        else if(iz==0) wbarbar_zl = wbar_zr;
        else wbarbar_zl = hx*hy*( 9*(w( ZL(in))+w(YL( ZL(in)))) - (w(YLL( ZL(in)))+w(YR( ZL(in)))) );
        if(iz==NZ-2) wbarbar_zr = 0;
        else if(iz==NZ-1) wbarbar_zr = wbar_zl;
        else wbarbar_zr = hx*hy*( 9*(w(ZRR(in))+w(YL(ZRR(in)))) - (w(YLL(ZRR(in)))+w(YR(ZRR(in)))) );
        real v_zr = iz==NZ-1 ? real(0) : v(in)+v(ZR(in));
        real v_zrr = v(in) + (iz==NZ-3 ? -v(ZRR(in)) : iz==NZ-2 ? real(0) : iz==NZ-1 ? v(ZLL(in)) : v(ZRRR(in)));
        real v_zl = iz==0 ? real(0) : v(ZL(in))+v(in);
        real v_zll = v(in) + (iz==2 ? -v(ZLL(in)) : iz==1 ? real(0) : iz==0 ? v(ZRR(in)) : v(ZLLL(in)));
        sumflux[1] += wbar_zr*v_zr - C1_27*wbarbar_zr*v_zrr -
                      wbar_zl*v_zl + C1_27*wbarbar_zl*v_zll;
    }

    if(iz!=0){ // for iz=0, there is no such variable as the velocity component in Z
    {
        real ubar_xr = 9*(u(XR(in))*DZ[iz] + u(ZL(XR(in)))*DZ[iz-1]);
        ubar_xr -= (iz==1) ? -u(ZL(XR(in)))*DZ[0] : u(ZLL(XR(in)))*DZ[iz-2];
        ubar_xr -= (iz==NZ-1) ? -u(XR(in))*DZ[NZ-1] : u(ZR(XR(in)))*DZ[iz+1];
        ubar_xr *= hy;
        real ubar_xl = 9*(u((in))*DZ[iz]+u(ZL((in)))*DZ[iz-1]);
        ubar_xl -= (iz==1) ? -u(ZL((in)))*DZ[0] : u(ZLL((in)))*DZ[iz-2];
        ubar_xl -= (iz==NZ-1) ? -u((in))*DZ[NZ-1] : u(ZR((in)))*DZ[iz+1];
        ubar_xl *= hy;
        real ubarbar_xr = 9*(u(XRR(in))*DZW[iz] + u(ZL(XRR(in)))*DZW[iz-1]);
        ubarbar_xr -= (iz==1) ? -u(ZL(XRR(in)))*DZW[0] : u(ZLL(XRR(in)))*DZW[iz-2];
        ubarbar_xr -= (iz==NZ-1) ? -u(XRR(in))*DZW[NZ-1] : u(ZR(XRR(in)))*DZW[iz+1];
        ubarbar_xr *= hy;
        real ubarbar_xl = 9*(u(XL(in))*DZW[iz] + u(ZL(XL(in)))*DZW[iz-1]);
        ubarbar_xl -= (iz==1) ? -u(ZL(XL(in)))*DZW[0] : u(ZLL(XL(in)))*DZW[iz-2];
        ubarbar_xl -= (iz==NZ-1) ? -u(XL(in))*DZW[NZ-1] : u(ZR(XL(in)))*DZW[iz+1];
        ubarbar_xl *= hy;
        sumflux[2] += ubar_xr*(w(in)+w(XR(in))) - C1_27*ubarbar_xr*(w(in)+w(XRRR(in))) -
                      ubar_xl*(w(XL(in))+w(in)) + C1_27*ubarbar_xl*(w(XLLL(in))+w(in));
    }
    {
        real vbar_yr = 9*(v(YR(in))*DZ[iz] + v(ZL(YR(in)))*DZ[iz-1]);
        vbar_yr -= (iz==1) ? -v(ZL(YR(in)))*DZ[0] : v(ZLL(YR(in)))*DZ[iz-2];
        vbar_yr -= (iz==NZ-1) ? -v(YR(in))*DZ[NZ-1] : v(ZR(YR(in)))*DZ[iz+1];
        vbar_yr *= hy;
        real vbar_yl = 9*(v((in))*DZ[iz]+v(ZL((in)))*DZ[iz-1]);
        vbar_yl -= (iz==1) ? -v(ZL((in)))*DZ[0] : v(ZLL((in)))*DZ[iz-2];
        vbar_yl -= (iz==NZ-1) ? -v((in))*DZ[NZ-1] : v(ZR((in)))*DZ[iz+1];
        vbar_yl *= hy;
        real vbarbar_yr = 9*(v(YRR(in))*DZW[iz] + v(ZL(YRR(in)))*DZW[iz-1]);
        vbarbar_yr -= (iz==1) ? -v(ZL(YRR(in)))*DZW[0] : v(ZLL(YRR(in)))*DZW[iz-2];
        vbarbar_yr -= (iz==NZ-1) ? -v(YRR(in))*DZW[NZ-1] : v(ZR(YRR(in)))*DZW[iz+1];
        vbarbar_yr *= hy;
        real vbarbar_yl = 9*(v(YL(in))*DZW[iz] + v(ZL(YL(in)))*DZW[iz-1]);
        vbarbar_yl -= (iz==1) ? -v(ZL(YL(in)))*DZW[0] : v(ZLL(YL(in)))*DZW[iz-2];
        vbarbar_yl -= (iz==NZ-1) ? -v(YL(in))*DZW[NZ-1] : v(ZR(YL(in)))*DZW[iz+1];
        vbarbar_yl *= hy;
        sumflux[2] += vbar_yr*(w(in)+w(YR(in))) - C1_27*vbarbar_yr*(w(in)+w(YRRR(in))) -
                      vbar_yl*(w(YL(in))+w(in)) + C1_27*vbarbar_yl*(w(YLLL(in))+w(in));
    }
    {
        real wzl = w(ZL(in));
        real wzr = iz==NZ-1 ? real(0) : w(ZR(in));
        real wzll = w(iz==1 ? in : ZLL(in));
        real wzrr = iz==NZ-2 ? real(0) : iz==NZ-1 ? w(in) : w(ZRR(in));
        real wzlll = iz==1 ? w(ZR(in)) : iz==2 ? w(ZL(in)) : w(ZLLL(in));
        real wzrrr = iz==NZ-3 ? real(0) : iz==NZ-2 ? w(ZR(in)) : iz==NZ-1 ? w(ZL(in)) : w(ZRRR(in));
        real wbar_zl = hx*hy*( 9*(wzl+w(in)) - (wzll+wzr) );
        real wbar_zr = hx*hy*( 9*(w(in)+wzr) - (wzl+wzrr) );
        real wbarbar_zl = hx*hy*( 9*(wzll+wzl) - (wzlll+w(in)) );
        real wbarbar_zr = hx*hy*( 9*(wzr+wzrr) - (w(in)+wzrrr) );
        sumflux[2] += wbar_zr*(w(in)+wzr) - wbar_zl*(wzl+w(in));
        /*
        // This is [V&V, p. 358, the formula below (41)]. No motivation behind it, and it does not work
        if(iz<NZ-2)       sumflux[2] -= C1_27*wbarbar_zr*(w(in)+wzrrr);
        else if(iz==NZ-2) sumflux[2] -= C1_27*hx*hy*(w(in)+w(ZR(in)))*(-w(ZR(in))+w(in));
        else              sumflux[2] -= C1_27*hx*hy*(w(in)+w(ZL(in)))*( w(ZL(in))+w(in));
        if(iz>2)          sumflux[2] += C1_27*wbarbar_zl*(wzlll+w(in));
        else if(iz==2)    sumflux[2] += C1_27*hx*hy*(w(in)+w(ZL(in)))*(-w(ZL(in))+w(in));
        else              sumflux[2] += C1_27*hx*hy*(w(in)+w(ZR(in)))*( w(ZR(in))+w(in));
        */
        if(iz<NZ-2)       sumflux[2] -= C1_27*wbarbar_zr*(w(in)+wzrrr);
        else if(iz==NZ-2) sumflux[2] -= C1_27*wbarbar_zr*(-w(ZR(in))+w(in));
        else              sumflux[2] -= C1_27*wbarbar_zr*( w(ZL(in))+w(in));
        if(iz>2)          sumflux[2] += C1_27*wbarbar_zl*(wzlll+w(in));
        else if(iz==2)    sumflux[2] += C1_27*wbarbar_zl*(-w(ZL(in))+w(in));
        else              sumflux[2] += C1_27*wbarbar_zl*( w(ZR(in))+w(in));    }
    }

    real minus_inv_vol = real(-1)/(32*(DZ[iz]-C1_9*DZW[iz])*hx*hy);
    sumflux[0] *= minus_inv_vol;
    sumflux[1] *= minus_inv_vol;
    // 1_32 results from:
    //      1/8 is the denominator (9/8 and -1/8 instead of 9 and -1)
    //      1/2 is due to the averaging in the expressions for the mass flux
    //      1/2 is due to the averaging in the expressions for variables being convected
    sumflux[2] *= (-real(1./32.)*inv_hx*inv_hy*GCZ[iz]);
    return sumflux;
}
#endif

template<>
__DEVICE__ real3 data::ViscFlux<tViscApproximation::CROSS>(int in, const real3* vel) const{
    int iz = in/NXY;
    real inv_hz = 1/DZ[iz];
    real inv_hz_alt = real(2)/(DZ[iz]+(iz==0 ? real(0) : DZ[iz-1]));
    real3 sumflux;
    sumflux += inv_hx*inv_hx*(vel[XR(in)]+vel[XL(in)]-2.*vel[in]);
    sumflux += inv_hy*inv_hy*(vel[YR(in)]+vel[YL(in)]-2.*vel[in]);
    if(iz!=NZ-1){
        real inv_hz_up = real(2)/(DZ[iz]+DZ[iz+1]);
        sumflux[0] += (u(ZR(in))-u(in))*inv_hz_up*inv_hz;
        sumflux[1] += (v(ZR(in))-v(in))*inv_hz_up*inv_hz;
        sumflux[2] += (w(ZR(in))-w(in))*inv_hz*inv_hz_alt;
    }
    else{
        real inv_hz_up = real(2)/(DZ[iz]);
        sumflux[0] += (-u(in))*inv_hz_up*inv_hz;
        sumflux[1] += (-v(in))*inv_hz_up*inv_hz;
        sumflux[2] += (-w(in))*inv_hz*inv_hz_alt;
    }
    if(iz!=0){
        sumflux[0] += (u(ZL(in))-u(in))*inv_hz_alt*inv_hz;
        sumflux[1] += (v(ZL(in))-v(in))*inv_hz_alt*inv_hz;
        sumflux[2] += (w(ZL(in))-w(in))*(1/DZ[iz-1])*inv_hz_alt;
    }
    else{
        sumflux[0] += (-u(in))*inv_hz_alt*inv_hz;
        sumflux[1] += (-v(in))*inv_hz_alt*inv_hz;
    }
    return visc*sumflux;
}

// Total viscosity (molecular + turbulent)
#if ENABLE_TURB_VISC
    #define mu(X) (visc+nu_t[X])
#else
    #define mu(X) visc
#endif

// Notation in this function:
// s?? = ??-component of the stress tensor
// 000 = center of cell `in`
// ?00 = ? half-cells to the right from cell `in` (if ? < 5) or (10-?) half-cells to the left (if ? > 5)
//       For instance, the location of vel[in][0] is 900
// 0?0 = same in Y direction
// 00? = same in Z direction
template<>
__DEVICE__ real3 data::ViscFlux<tViscApproximation::NAIVE>(int in, const real3* vel) const{
    int iz = in/NXY;
    real inv_hz      = 1/DZ[iz];
    real inv_hz_up   = 2/(DZ[iz] + (iz==NZ-1 ? real(0) : DZ[iz+1]));
    real inv_hz_down = 2/(DZ[iz] + (iz==0    ? real(0) : DZ[iz-1]));

    real3 sumflux;
    real sxx_000 = 2*(u(XR(in))-u(in))*inv_hx * mu(in);
    real sxx_800 = 2*(u(in)-u(XL(in)))*inv_hx * mu(XL(in));
    sumflux[0] += (sxx_000-sxx_800)*inv_hx;
    real sxy_910 = ( (u(YR(in))-u(in))*inv_hy + (v(YR(in))-v(XL(YR(in))))*inv_hx ) * 0.25f*(mu(in)+mu(XL(in))+mu(YR(in))+mu(XL(YR(in))));
    real sxy_990 = ( (u(in)-u(YL(in)))*inv_hy +         (v(in)-v(XL(in)))*inv_hx ) * 0.25f*(mu(YL(in))+mu(YL(XL(in)))+mu(in)+mu(XL(in)));
    sumflux[0] += (sxy_910-sxy_990)*inv_hy;
    real sxz_901 = (iz==NZ-1) ? (-2*u(in)*inv_hz*visc) :
                   ( (u(ZR(in))-u(in))*inv_hz_up   + (w(ZR(in))-w(XL(ZR(in))))*inv_hx ) * 0.25f*(mu(in)+mu(XL(in))+mu(ZR(in))+mu(XL(ZR(in))));
    real sxz_909 = (iz==0) ? (2*u(in)*inv_hz*visc) :
                   ( (u(in)-u(ZL(in)))*inv_hz_down +         (w(in)-w(XL(in)))*inv_hx ) * 0.25f*(mu(ZL(in))+mu(ZL(XL(in)))+mu(in)+mu(XL(in)));
    sumflux[0] += (sxz_901-sxz_909)*inv_hz;

    real sxy_190 = ( (u(XR(in))-u(XR(YL(in))))*inv_hy + (v(XR(in))-v(in))*inv_hx ) * 0.25f*(mu(in)+mu(XR(in))+mu(YL(in))+mu(XR(YR(in))));
  //real sxy_990 = ( (u(in)-u(YL(in)))*inv_hy +         (v(in)-v(XL(in)))*inv_hx ) * 0.25f*(mu(YL(in))+mu(YL(XL(in)))+mu(in)+mu(XL(in))); // already defined
    sumflux[1] += (sxy_190-sxy_990)*inv_hx;
    real syy_000 = 2*(v(YR(in))-v(in))*inv_hy * mu(in);
    real syy_080 = 2*(v(in)-v(YL(in)))*inv_hy * mu(YL(in));
    sumflux[1] += (syy_000-syy_080)*inv_hy;
    real syz_091 = (iz==NZ-1) ? (-2*v(in)*inv_hz*visc) :
                   ( (v(ZR(in))-v(in))*inv_hz_up   + (w(ZR(in))-w(YL(ZR(in))))*inv_hy ) * 0.25f*(mu(in)+mu(YL(in))+mu(ZR(in))+mu(YL(ZR(in))));
    real syz_099 = (iz==0) ? (2*v(in)*inv_hz*visc) :
                   ( (v(in)-v(ZL(in)))*inv_hz_down +         (w(in)-w(YL(in)))*inv_hy ) * 0.25f*(mu(ZL(in))+mu(ZL(YL(in)))+mu(in)+mu(YL(in)));
    sumflux[1] += (syz_091-syz_099)*inv_hz;

    if(iz==0) return sumflux; // no DOF for Z-velocity at iz=0
    real sxz_109 = ( (u(XR(in))-u(XR(ZL(in))))*inv_hz_down + (w(XR(in))-w(in))*inv_hx ) * 0.25f*(mu(ZL(in))+mu(ZL(XR(in)))+mu(in)+mu(XR(in)));
  //real sxz_909 = ...; // already defined
    sumflux[2] += (sxz_109 - sxz_909)*inv_hx;
    real syz_019 = ( (v(YR(in))-v(YR(ZL(in))))*inv_hz_down + (w(YR(in))-w(in))*inv_hy ) * 0.25f*(mu(ZL(in))+mu(ZL(YR(in)))+mu(in)+mu(YR(in)));
  //real syz_099 = ...; // already defined
    sumflux[2] += (syz_019 - syz_099)*inv_hy;
    real szz_000 = 2*((iz==NZ-1 ? real(0) : w(ZR(in))) - w(in))*inv_hz * mu(in);
    real szz_008 = 2*(w(in) - (iz==0 ? real(0) : w(ZL(in)))) * mu(ZL(in)) / DZ[iz-1];
    sumflux[2] += (szz_000 - szz_008)*inv_hz_down;

    // Renormalization - for a definition of volumes, consistent with the convection
    //sumflux[0] *= 0.125*(9*DZ[iz] - DZW[iz])*inv_hz;
    //sumflux[1] *= 0.125*(9*DZ[iz] - DZW[iz])*inv_hz;
    return sumflux;
}
#undef mu
#undef u
#undef v
#undef w


void data::CalcFluxTerm_FD(const real3* u, int DoConv, int DoVisc, real3* kuhat) const{
    Forall([u,DoConv,DoVisc,kuhat] __DEVICE__ (int in){
        real3 sumflux;
        if(DoConv) sumflux += D.ConvFlux(in, u);
        if(DoVisc) sumflux += D.ViscFlux<VISC_APPR_TYPE>(in, u);
        if(in<NXY) sumflux[2] = 0; // just for a case
        kuhat[in] = sumflux;
    });
}

void data::ExplicitTerm(const real3* u, real3* kuhat) const{
    t_ExplicitTerm.beg();
    Forall([u,kuhat] __DEVICE__ (int in){
        kuhat[in] = real3(D.sour_dpdx,0,0) + D.ConvFlux(in,u) + D.ViscFlux<VISC_APPR_TYPE>(in,u);
        if(in<NXY) kuhat[in][2] = 0; // just for a case
    });
    t_ExplicitTerm.end();
}


// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

// Enforce the zero pressure average
void data::NormalizePressure(real* f) const{
    t_PressureSolver[3].beg();
    thrust::counting_iterator<int> I;
    #if(CONV_APPR_TYPE==CONV_HW)
    real psum = thrust::transform_reduce(THRUST_POLICY, I, I+N3D, [f] __DEVICE__ (int i)->real {
        return f[i]*D.DZ[i/NXY];
        }, 0., thrust::plus<real>());
    real p_shift = psum / ((ZZ[NZ]-ZZ[0])*NXY);
    Forall1D([f,p_shift] __DEVICE__ (int i){ f[i] -= p_shift; });
    #endif
    #if(CONV_APPR_TYPE==CONV_VV)
    real psum = thrust::transform_reduce(THRUST_POLICY, I, I+N3D, [f] __DEVICE__ (int i)->real {
        return f[i]*D.KERG[i/NXY]*D.DZ[i/NXY];
        }, 0., thrust::plus<real>());
    real p_shift = psum / (KERG_norm2*NXY);
    Forall1D([f,p_shift] __DEVICE__ (int i){ f[i] -= p_shift*D.KERG[i/NXY]; });
    #endif
    t_PressureSolver[3].end();
}

// Integral values
real data::CalcKineticEnergy(const real3* u) const{
    thrust::counting_iterator<int> I;
    return 0.5*thrust::transform_reduce(THRUST_POLICY, I, I+N3D, [u] __DEVICE__ (int i)->real {
        return ((u[i][0]*u[i][0]+u[i][1]*u[i][1])*D.DZ[i/NXY] + u[i][2]*u[i][2]*(i<NXY ? real(0) : real(0.5)*(D.DZ[i/NXY]+D.DZ[i/NXY-1])))*D.hx*D.hy;
        }, real(0), thrust::plus<real>());
}

real3 data::CalcIntegral(const real3* u) const{
    thrust::counting_iterator<int> I;
    return thrust::transform_reduce(THRUST_POLICY, I, I+N3D, [u] __DEVICE__ (int i)->real3 {
        return real3(u[i][0]*D.DZ[i/NXY], u[i][1]*D.DZ[i/NXY], i<NXY ? real(0) : u[i][2]*real(0.5)*(D.DZ[i/NXY]+D.DZ[i/NXY-1]))*D.hx*D.hy;
        }, real3(), thrust::plus<real3>());
}

__DEVICE__ real data::CalcCFL_loc(int i, const real3* u, double tau, int IsConvVisc) const{
    // Convection
    real inv_hz = 1/DZ[i/NXY];
    real CFL = 0;
    if(IsConvVisc&1){
        CFL += tau*(fabs(u[i][0])*inv_hx + fabs(u[i][1])*inv_hy + fabs(u[i][2])*inv_hz);
    }
    if(IsConvVisc&2){
        double visc_coeff = nu_t!=NULL ? visc+nu_t[i] : visc;
        CFL += 4*((VISC_APPR_TYPE!=tViscApproximation::CROSS)+1)*tau*visc_coeff*(inv_hx*inv_hx + inv_hy*inv_hy + inv_hz*inv_hz);
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
__global__ void CalcVelocityProfile_kernel(int NZ, const real3* vel, const real* p, const real* nu_t){
    __shared__ float _shm[VELPROF_BLK*sizeof(real9<real>)/4];
    real9<real>* shm = (real9<real>*)_shm;

    int iz = blockIdx.x;
    if(iz>=NZ) return; // should not happen

    real9<real> sum;
    if(threadIdx.x<VELPROF_BLK){
        for(int in=iz*NXY + threadIdx.x; in<(iz+1)*NXY; in+=VELPROF_BLK){
            real u = vel[in][0];
            real v = vel[in][1];
            real w = real(0.5)*(vel[in][2]+(iz==NZ-1 ? real(0) : vel[ZR(in)][2]));
            sum.u[0] += u;
            sum.u[1] += v;
            sum.u[2] += w;
            sum.uu[0] += u*u;
            sum.uu[1] += v*v;
            sum.uu[2] += w*w;
            sum.uu[3] += u*v;
            sum.uu[4] += u*w;
            sum.uu[5] += v*w;
            if(nu_t!=NULL) sum.nu_t += nu_t[in];
            sum.p += p[in];
            sum.pp += p[in]*p[in];
        }
        sum *= real(1)/NXY;
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

void data::CalcProfiles(const real3* u, const real* p){
    const int BLK = VELPROF_BLK;
    const int GRID = NZ;
    CalcVelocityProfile_kernel<<<GRID,BLK>>>(NZ,u,p,nu_t);
    GPU_CHECK_CRASH(cudaGetLastError());
    #ifdef SYNC_AFTER_EACH_KERNEL
        GPU_CHECK_CRASH(cudaDeviceSynchronize());
    #endif
}
#else
void data::CalcProfiles(const real3* vel, const real* p){
    Forall1D([vel,p] __DEVICE__ (int iz){
        real9<real> sum;
        for(int in=iz*NXY; in<(iz+1)*NXY; in++){
            real u = vel[in][0];
            real v = vel[in][1];
            real w = real(0.5)*(vel[in][2]+(iz==D.NZ-1 ? real(0) : vel[ZR(in)][2]));
            sum.u[0] += u;
            sum.u[1] += v;
            sum.u[2] += w;
            sum.uu[0] += u*u;
            sum.uu[1] += v*v;
            sum.uu[2] += w*w;
            sum.uu[3] += u*v;
            sum.uu[4] += u*w;
            sum.uu[5] += v*w;
            if(D.nu_t!=NULL) sum.nu_t += D.nu_t[in];
            sum.p += p[in];
            sum.pp += p[in]*p[in];
        }
        sum *= real(1)/(NXY);
        D.av_u[iz] = sum;
    }, NZ);
}
#endif


// ---------------------------------------------------------------------------
// Turbulence
// ---------------------------------------------------------------------------

void data::RecalcTurbVisc(const real3* u){
    if(nu_t==NULL) return;
    Forall([u] __DEVICE__ (int i){ D.RecalcTurbVisc(i, u); });
}

#define u(X) vel[(X)][0]
#define v(X) vel[(X)][1]
#define w(X) vel[(X)][2]
__DEVICE__ real data::CalcAbsS(int in, const real3* vel){
    int iz = in/NXY;
    real inv_hz      = 1/DZ[iz];
    real inv_hz_up   = 2/(DZ[iz] + (iz==NZ-1 ? real(0) : DZ[iz+1]));
    real inv_hz_down = 2/(DZ[iz] + (iz==0    ? real(0) : DZ[iz-1]));

    // Diagonal components of the strain tensor - at the cell center
    real AbsS = 0;
    {
        real sxx = (u(XR(in))-u(in))*inv_hx;
        real syy = (v(YR(in))-v(in))*inv_hy;
        real szz = ((iz==NZ-1 ? real(0) : w(ZR(in))) - w(in))*inv_hz;
        AbsS = sxx*sxx + syy*syy + szz*szz;
    }
    // Non-diagonal components of the symmetric part of the strain tensor - at edge centers, need to average
    // 0.125 = 0.25 (taking the symmetric part, squared) * 0.25 (averaging by four edges) * 2 (two elements of S^T S)
    {
        real sxy_910 = (u(YR(in))-u(in))*inv_hy + (v(YR(in))-v(XL(YR(in))))*inv_hx;
        real sxy_990 = (u(in)-u(YL(in)))*inv_hy +         (v(in)-v(XL(in)))*inv_hx;
        real sxy_110 = (u(XR(YR(in)))-u(XR(in)))*inv_hy + (v(XR(YR(in)))-v(YR(in)))*inv_hx;
        real sxy_190 = (u(XR(in))-u(XR(YL(in))))*inv_hy +         (v(XR(in))-v(in))*inv_hx;
        AbsS += 0.125f*(sxy_910*sxy_910 + sxy_990*sxy_990 + sxy_110*sxy_110 + sxy_190*sxy_190);
    }
    {
        real sxz_901 = (iz==NZ-1) ? (-2*u(in)*inv_hz) :
                       (u(ZR(in))-u(in))*inv_hz_up   + (w(ZR(in))-w(XL(ZR(in))))*inv_hx;
        real sxz_909 = (iz==0) ? (2*u(in)*inv_hz) :
                       (u(in)-u(ZL(in)))*inv_hz_down +         (w(in)-w(XL(in)))*inv_hx;
        real sxz_101 = (iz==NZ-1) ? (-2*u(XR(in))*inv_hz) :
                       (u(XR(ZR(in)))-u(XR(in)))*inv_hz_up   + (w(XR(ZR(in)))-w(ZR(in)))*inv_hx;
        real sxz_109 = (iz==0) ? (2*u(XR(in))*inv_hz) :
                       (u(XR(in))-u(ZL(XR(in))))*inv_hz_down +         (w(XR(in))-w(in))*inv_hx;
        AbsS += 0.125f*(sxz_901*sxz_901 + sxz_909*sxz_909 + sxz_101*sxz_101 + sxz_109*sxz_109);
    }
    {
        real syz_091 = (iz==NZ-1) ? (-2*v(in)*inv_hz) :
                       (v(ZR(in))-v(in))*inv_hz_up   + (w(ZR(in))-w(YL(ZR(in))))*inv_hy;
        real syz_099 = (iz==0) ? (2*v(in)*inv_hz) :
                       (v(in)-v(ZL(in)))*inv_hz_down +         (w(in)-w(YL(in)))*inv_hy;
        real syz_011 = (iz==NZ-1) ? (-2*v(YR(in))*inv_hz) :
                       (v(YR(ZR(in)))-v(YR(in)))*inv_hz_up   + (w(YR(ZR(in)))-w(ZR(in)))*inv_hy;
        real syz_019 = (iz==0) ? (2*v(YR(in))*inv_hz) :
                       (v(YR(in))-v(ZL(YR(in))))*inv_hz_down +         (w(YR(in))-w(in))*inv_hy;
        AbsS += 0.125f*(syz_091*syz_091 + syz_099*syz_099 + syz_011*syz_011 + syz_019*syz_019);
    }
    AbsS=sqrt(2*AbsS);
    return AbsS;
}
#undef u
#undef v
#undef w

__DEVICE__ void data::RecalcTurbVisc(int in, const real3* vel){
    int iz = in/NXY;
    const real H = 0.5f*(ZZ[NZ] - ZZ[0]);
    const real Z = 0.5f*(ZZ[iz]+ZZ[iz+1]) - ZZ[0];
    const real dist_to_wall = (Z<H) ? Z : 2*H-Z;
    const real y_plus = dist_to_wall*inv_length_scale;

    const real C_SMAG = 0.1;
    const real C_exp = 0.04*0.04*0.04;
    const real AbsS = CalcAbsS(in, vel);
    const real f_VD = 1-exp(-C_exp*y_plus*y_plus*y_plus); // van Driest multiplier
    const real V = hx*hy*DZ[iz];
    real Delta_LES = C_SMAG*pow(V,C1_3);
    if(Delta_LES>real(0.41)*dist_to_wall) Delta_LES=real(0.41)*dist_to_wall;
    nu_t[in] = f_VD*Delta_LES*Delta_LES*AbsS;
}


// ---------------------------------------------------------------------------
// General methods
// ---------------------------------------------------------------------------

void data::Step(double t_start, double tau, int current_field, int bdf_order){
    if(bdf_order<1 || bdf_order>3) { printf("Wrong bdf_order\n"); exit(0); }
    int next_field = (current_field+1)%4;
    const real inv_tau = real(1./tau);
    const real bd0=(bdf_order==3) ? real(1.5+C1_3) : (bdf_order==2) ? real(1.5) : real(1.0);
    const real tau_over_bd0 = tau / bd0;

    if(nu_t!=NULL) RecalcTurbVisc(vel[current_field]);

    // Explicit fluxes at t_n
    ExplicitTerm(vel[current_field], flux[current_field]);

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

    // Pressure solver
    UnapplyGradient(flux[next_field], pres[next_field]);

    // Velocity correction
    ApplyGradient(pres[next_field], vel[next_field]);
    Forall1D([tau_over_bd0,next_field] __DEVICE__ (int in){
        D.vel[next_field][in] = (D.flux[next_field][in] - D.vel[next_field][in]) * tau_over_bd0;
    });
}

void data::alloc_all(){
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
    nu_t = NULL;
    #if ENABLE_TURB_VISC
        nu_t = alloc<real>(N3D);
    #endif
    cosphi[0]   = alloc<double>(NX);
    cosphi[1]   = alloc<double>(NY);
    av_u        = alloc< real9<real> >(NZ);
}
void data::dealloc_all(){
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
