#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <omp.h>

#include "data_global.h"
#include "mytimer.h"
static const double Pi = 3.14159265358979323846;
MyTimer t_ExplicitTerm, t_ApplyGradient, t_PressureSolver[4];

// Fills the matrix of the Laplace operator that results from the VV4 approximation
void data_ext::FillMatrix(real* M){
    #if(CONV_APPR_TYPE==CONV_VV)
    const real* inv_hz = GCZ;
    #else
    const real* inv_hz = NULL; // this subroutine is not called
    #endif
    const int w = 3;
    const int NB = 2*w+1;
    const real grad_coef[4] = {1./27., -1., 1., -1./27.};

    for(int j=0; j<NZ; j++){ // cell index related to the row
        for(int pos=0; pos<NB; pos++) M[j*NB + pos] = 0.;

        for(int ll=-1; ll<=2; ll++){
            real coeff1 = grad_coef[ll+1];

            int l = j+ll; // node index where the gradient is defined
            if(l==0 || l==NZ) continue; // mass flux across the boundary is zero
            if(l<0) l=-l; // mass flux for l=-1 is defined by symmetry (w. r. t. boundary node)
            if(l>NZ) l=2*NZ-l; // mass flux for l=NZ+1 is defined by symmetry

            for(int mm=-2; mm<=1; mm++){
                real coeff2 = grad_coef[mm+2];

                int m = l+mm; // cell index
                if(m<0) { m=-m-1; coeff2*=-1.; } // strange anti-symmetry w. r. t. boundary node for the pressure
                if(m>=NZ) { m=2*NZ-m-1; coeff2*=-1.; }

                int pos = m-j+w;
                if(pos<0 || pos>=NB || l<=0 || l>=NZ) {
                    printf("Internal error\n"); exit(0);
                }
                M[j*NB + pos] += coeff1 * coeff2 * inv_hz[l];
            }
        }
    }
}


int IsPowerOfTwo(int i){
    if(i<=0) return 0;
    while(i>1){ if(i&1) return 0; i>>=1; };
    return 1;
}

// Get the point where variable `ivar` is defined. ivar=0 for u, ivar=1 for v, ivar=2 for w, ivar=3 for p
real3 data_ext::GetCoor(int in, int ivar) const{
    if(in<0 || in>=N3D) return real3();
    int i[3] = {in%NX, (in/NX)%NY, in/(NX*NY)};
    real3 c((i[0]+0.5)*hx, (i[1]+0.5)*hy, 0.5*(ZZ[i[2]] + ZZ[i[2]+1]));
    if(ivar==0) c[0] = i[0]*hx;
    if(ivar==1) c[1] = i[1]*hy;
    if(ivar==2) c[2] = ZZ[i[2]];
    return c;
}

// Solving a system with a banded matrix using the Gauss elimination without pivoting
// Matrix format: N lines of size NB each. NB must be odd
// Elements are written left to right, so the diagonal elements are NB/2, NB+NB/2, ..., (N-1)*NB+NB/2
// Attention! Matrix is changed during the process
// The solution is written to the same address as the right-hand side
template<typename fpv>
int solve_banded(int N, int NB, fpv* M, fpv* x, fpv tiny){
    if(!(NB&1)) return 1; // NB must be odd
    const int w = NB/2; // number of lower diagonals (and upper diagonals as well)
    // Forward substitution
    for(int i=0; i<N; i++){
        // Substracting the previous lines
        for(int j=-w; j<0; j++){ // j<0 is the offset to the previous line
            if(i+j<0) continue; // no such line
            fpv mult = -M[i*NB + w+j];
            for(int k=0; k<=w; k++){
                if(i+j+k>=N) continue; // no such column
                M[i*NB + (w+j)+k] += mult*M[(i+j)*NB + w+k]; // line(i) += mult*line(i+j)
            }
            x[i] += mult*x[i+j];
        }
        // Normalizing the line
        if(fabs(M[i*NB + w]) < tiny) return 2; // Dividing by zero
        {
            fpv mult = fpv(1)/M[i*NB + w];
            for(int k=0; k<=w; k++){
                if(i+k>=N) continue; // no such column
                M[i*NB + w+k] *= mult;
            }
            x[i] *= mult;
        }
    }
    // Backward substitution
    for(int i=N-1; i>=0; i--){
        for(int j=1; j<=w; j++){ // j>0 is the offset to the line below
            if(i+j>=N) continue;
            x[i] -= M[i*NB+w+j]*x[i+j];
            // i-th line of the matrix is replaced by the i-th line of the unit matrix. No need to write this to M
        }
    }
    return 0;
}

// Having the mesh information (Z coordinates), evaluate all derivative data that will be copied to the device
void data_ext::InitDerivativeMeshData(){
    for(int i=0; i<NZ; i++){
        DZ[i] = ZZ[i+1]-ZZ[i];
    }

    #if(CONV_APPR_TYPE==CONV_HW)
    MAT_A[0]=0.; // unused
    for(int j=1; j<=NZ-1; j++) MAT_A[j]=2./(DZ[j-1]+DZ[j]);
    #endif

    #if(CONV_APPR_TYPE==CONV_VV)
    for(int iz=0; iz<NZ; iz++){
        DZW[iz] = (DZ[iz] + DZ[iz==0 ? iz:iz-1] + DZ[iz==NZ-1 ? iz:iz+1]) / 3.;
        GC1[iz] =  DZ[iz]           / (DZ[iz] - DZW[iz]/9.);
        GC2[iz] = -(1./27.)*DZW[iz] / (DZ[iz] - DZW[iz]/9.);
        GCZ[iz] = 1 / (0.5*(DZ[iz]+DZ[iz-1]) - (1./27.)*(DZ[iz]+DZ[iz-1]+0.5*(DZ[iz==NZ-1?iz:iz+1]+DZ[iz==1?0:iz-2])));
    }

    // lower diagonals of the matrix of the pressure system
    real MAT[7*NZ_MAX];
    FillMatrix(MAT); // this gives all 7 diagonals of the matrix
    for(int iz=0; iz<NZ; iz++){ // preserve lower diagonals only. Set out-of-bounds elements to zero
        for(int i=0; i<3; i++) MAT_A[iz*3+i] = (iz+i-3>=0) ? MAT[iz*7+i] : real(0);
    }
    // Some diagonal elements
    MAT_D[0] = MAT[3]; MAT_D[1] = MAT[7+3]; MAT_D[2] = MAT[7*2+3];
    MAT_D[3] = MAT[(NZ-3)*7+3]; MAT_D[4] = MAT[(NZ-2)*7+3]; MAT_D[5] = MAT[(NZ-1)*7+3];

    // kernel of the gradient operator
    const real eps = real(1./27.);
    MAT[1] = -1; MAT[2] = eps;
    for(int iz=1; iz<NZ-1; iz++){ MAT[iz*3+0] = MAT[iz*3+2] = eps; MAT[iz*3+1] = -1+eps; }
    MAT[(NZ-1)*3] = eps; MAT[(NZ-1)*3+1] = -1;
    real K[NZ_MAX];
    for(int iz=0; iz<NZ; iz++) K[iz]=1;
    int errcode = solve_banded(NZ, 3, MAT, K, real(1e-10));
    if(errcode) { printf("Internal error in solving system for K\n"); exit(0); }

    KERG_norm2 = 0;
    for(int iz=0; iz<NZ; iz++){
        KERG[iz] = K[iz];
        KERG_norm2 += K[iz]*K[iz]*DZ[iz];
    }
    #endif
}

void data_ext::SetInitialFields(int current_field, double u_tau){
    // Try reading input.vtk first
    int read_result = ReadData("input.vtk", vel[current_field], pres[current_field]);
    if(!read_result){ printf("Initial data read\n"); return; }

    // If there is no input.vtk, prescribe the data explicitly
    printf("input.vtk not found. Starting from the beginning\n");
    double Hx = hx*NX, Hy = hy*NY, Hz = ZZ[NZ]-ZZ[0], zmin = ZZ[0], H = 0.5*Hz;
    for(int in=0; in<N3D; ++in){
        real3 v;
        { // u
            real3 r = GetCoor(in, 0);
            double x = r[0], y = r[1], z = r[2];
            double zplus = std::min(z, Hz-z)*u_tau/visc;
            v[0] = 0.5*(z-zmin)*(zmin+Hz-z)*sour_dpdx/visc;
            v[0] -= cos(x)*sin(2.*Pi*y/Hy)*zplus*exp(-0.01*zplus*zplus);
        }
        { // v
            real3 r = GetCoor(in, 1);
            double x = r[0], y = r[1], z = r[2];
            double zplus = std::min(z, 2.*H-z)*u_tau/visc;
            v[1] += sin(x)*cos(2.*Pi*y/Hy)*zplus*exp(-0.01*zplus*zplus)/(2.*Pi/Hy);
        }

        vel[current_field][in]=v;
        pres[current_field][in]=0;
    }
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
    if(ENABLE_TURB_VISC==1 && VISC_APPR_TYPE==tViscApproximation::CROSS){ printf("tViscApproximation::CROSS does not work for a turbulence model\n"); exit(0); }
    if(PM.DeviceID<0 || PM.DeviceID>=DeviceCount) { printf("DEVICE ID OUT OF RANGE\n"); exit(0); }

    // Problem parameters
    const double Re_tau = 395; // u_tau*H/visc
    const double H = 1.; // channel half-height
    const double visc = 1./Re_tau; // kinematic viscosity (affects the velocity scale only)
    const double u_tau = Re_tau*visc; // expected friction velocity
    const double sour_dpdx = u_tau*u_tau / H; // source in the momentum equation per unit volume
    const double Hx = 4.*Pi*H, Hy = 2.*Pi*H, Hz = 2.*H; // domain size
    const double xmin = 0., ymin = -0.5*Hy, zmin = 0.; // offset, does not matter

    const int NZ = PM.NZ; // number of cells in Z
    if(NZ > NZ_MAX) { printf("WRONG NZ_MAX=%i (actual number of nodes = %i)\n", NZ_MAX, NZ); exit(0); }

    int USE_UNIFORM_MESH = 0;
    {
        using namespace data_cpu;
        if(USE_UNIFORM_MESH){ // uniform mesh - for the approximation check
            for(int i=0; i<=NZ; i++) D.ZZ[i] = i*(Hz/NZ);
        }
        else{
            const double stretch = 6.5;
            for(int i=0; i<=NZ/2; i++){
                D.ZZ[i] = H*sinh(stretch*i/NZ)/sinh(0.5*stretch);
                D.ZZ[NZ-i] = 2.*H - D.ZZ[i];
            }
        }
        for(int i=0; i<NZ; i++) D.ZZ[i] += zmin;

        // Misc parameters
        D.visc = visc;
        D.sour_dpdx = sour_dpdx;
        D.inv_length_scale = u_tau/visc;

        // Mesh parameters
        D.NZ = NZ;
        D.N3D = NX*NY*NZ;
        D.hx = Hx/NX;
        D.hy = Hy/NY;
        D.inv_hx = 1./D.hx;
        D.inv_hy = 1./D.hy;
        D.area = Hx*Hy*Hz;
        D.xmin = xmin;
        D.ymin = ymin;
    }

    #define DATA_CPU data_cpu::D
    DATA_CPU.InitDerivativeMeshData();

    // Data on host with device pointers
    struct data_cuda::data DATA_CUDA;
    // Copy the main parameters and the derivative mesh data to DATA_CUDA
    *(static_cast<main_params*>(&DATA_CUDA)) = *(static_cast<main_params*>(&DATA_CPU));

    // Alloc global memory on CPU
    DATA_CPU.alloc_all();
    DATA_CPU.InitFourier();

    // Preparing the device
    GPU_CHECK_CRASH( cudaSetDevice(PM.DeviceID) ); // Set the device
    DATA_CUDA.alloc_all(); // alloc memory on device
    void CopyDataToDevice(const data_cuda::data&);
    CopyDataToDevice(DATA_CUDA); // copy data (including the pointers just initialized) to the constant memory of the device

    DATA_CUDA.InitFourier();

    // Self-tests
    DATA_CPU.MakeCPUChecks(USE_UNIFORM_MESH); // Subroutines check (CPU only)
    DATA_CPU.MakeCPUGPUChecks(DATA_CUDA); // Comparison of specific subroutines on CPU and GPU

    // Initial data
    int current_field = 0;
    DATA_CPU.SetInitialFields(current_field, u_tau);

    #define WORK_ON_GPU // uncomment for CUDA solver

    #ifdef WORK_ON_GPU
        #define FlowSolver DATA_CUDA
        const int N3D = DATA_CPU.N3D;
    #else
        #define FlowSolver DATA_CPU
        omp_set_num_threads(1);
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

    std::vector< real9<double> > intergal_av_u(NZ); // always in double precision
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
        FlowSolver.CalcProfiles(FlowSolver.vel[current_field], FlowSolver.pres[current_field]);
        t_velprof.end();
        #ifdef WORK_ON_GPU
            t_velprof_copy.beg();
            GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.av_u, DATA_CUDA.av_u, NZ*sizeof(real9<real>), cudaMemcpyDeviceToHost ) );
            t_velprof_copy.end();
        #endif
        if(t>=PM.TimeStartAveraging){
            for(int iz=0; iz<NZ; iz++) intergal_av_u[iz] += real(tau)*DATA_CPU.av_u[iz];
            intergal_time += tau;
        }

        t_output.beg();
        t += tau;
        if(t>=PM.TimeMax || iTimeStep+1==PM.TimeStepsMax) isFinalStep=1;
        double dudy_current = (DATA_CPU.av_u[0].u[0]-0.)/(0.5*DATA_CPU.DZ[0]);
        double u_fric_current = dudy_current<0. ? 0. : sqrt(dudy_current*visc);
        double dudy_integral = intergal_time>0. ? intergal_av_u[0].u[0]/(0.5*DATA_CPU.DZ[0]*intergal_time) : 0.;
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
            DATA_CPU.DumpData(fname, DATA_CPU.vel[current_field], DATA_CPU.pres[current_field], NULL);
        }
        if(DoWriteOutput2){
            char fname[256];
            sprintf(fname, "r%05i.dat", iTimeStep);
            DATA_CPU.DumpProfiles(DATA_CPU.av_u, fname);
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
    printf("Total time: %.04f\n", t_Total.timer);
    #ifdef SYNC_AFTER_EACH_KERNEL // without per-kernel syncronization, these channels make no sense
    printf("Channels: step %.04f, expl %.04f, grad %.04f, pres %.04f (F=%.04f T=%.04f N=%.04f), velprof %.04f (copy %.04f), reduction %.04f, output %.04f\n", t_Step.timer, t_ExplicitTerm.timer, t_ApplyGradient.timer, t_PressureSolver[0].timer, t_PressureSolver[1].timer, t_PressureSolver[2].timer, t_PressureSolver[3].timer, t_velprof.timer, t_velprof_copy.timer, t_reduction.timer, t_output.timer);
    #endif

    #ifdef WORK_ON_GPU
    GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.vel[current_field], DATA_CUDA.vel[current_field], N3D*sizeof(real3), cudaMemcpyDeviceToHost ) );
    GPU_CHECK_CRASH( cudaMemcpy( DATA_CPU.pres[current_field], DATA_CUDA.pres[current_field], N3D*sizeof(real), cudaMemcpyDeviceToHost ) );
    #endif
    DATA_CPU.DumpData("output.vtk", DATA_CPU.vel[current_field], DATA_CPU.pres[current_field], NULL);
    DATA_CPU.DumpProfiles(DATA_CPU.av_u, "output.dat");

    if(intergal_time>0.){
        for(int iz=0; iz<NZ; iz++) intergal_av_u[iz] *= 1./intergal_time;
        DATA_CPU.DumpProfiles(intergal_av_u.data(), "res.dat");
        //double dudy = 1.5*(intergal_av_u[1].u[0]-0.)/(data_cpu::DZ[0])-0.5*(intergal_av_u[2].u[0]-intergal_av_u[1].u[0])/(data_cpu::DZ[1]);
        double dudy = (intergal_av_u[0].u[0]-0.)/(0.5*DATA_CPU.DZ[0]);
        double u_fric = dudy<0. ? 0. : sqrt(visc*dudy);
        printf("u_fric: obtained=%.6f, expected=%.6f\n", u_fric, u_tau);
    }

    DATA_CPU.dealloc_all();
    DATA_CUDA.dealloc_all();
    return 0;
}
