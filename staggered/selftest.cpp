#include <stdio.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "data_global.h"
using namespace std;

// Exactness of the convective fluxes approximation on linear velocity fields (uniform meshes only)
// 1A - only one velocity component ('jtest') is linear (in coordinate 'jcoor'), others are zero
// 1B - all three velocity fields are linear
void data_ext::CheckConvectionConsistency(int print_all){
    if(print_all) printf("Explicit fluxes approximation check:\n");
    real max_err = 0;
    std::vector<real3> u(N3D), f(N3D);
    // 1A
    for(int jtest=0; jtest<3; jtest++)
    for(int jcoor=0; jcoor<3; jcoor++){
        for(int in=0; in<N3D; ++in){
            u[in] = real3();
            u[in][jtest] = GetCoor(in, jtest)[jcoor];
        }
        CalcFluxTerm_FD(u.data(), 1, 0, f.data()); // 1 - evaluate convection, 0 - do not evaluate viscosity
        real3 err;
        for(int in=0; in<N3D; ++in){
            int i[3] = {in%NX, (in/NX)%NY, in/(NX*NY)};
            const int BB = 6; // skip near-boundary strips of width BB (including periodic boundaries)
            if(i[0]<BB || i[0]>=NX-BB || i[1]<BB || i[1]>=NY-BB || i[2]<BB || i[2]>=NZ-BB) continue;
            real3 fex;
            if(jcoor==jtest) fex[jcoor] -= 2*GetCoor(in, jcoor)[jcoor];
            for(int icoor=0; icoor<3; icoor++){
                real errloc = fabs(f[in][icoor] - fex[icoor]);
                err[icoor] = std::max(err[icoor], errloc);
            }
        }
        if(print_all) printf("Test %i %i: e = %.0e %.0e %.0e\n", jtest, jcoor, err[0], err[1], err[2]);
        max_err = std::max(max_err, abs(err));
    }

    // 1B
    int itest[3]; // u = r[itest[0]], v = r[itest[1]], w = r[itest[2]]
    for(itest[0]=0; itest[0]<3; itest[0]++)
    for(itest[1]=0; itest[1]<3; itest[1]++)
    for(itest[2]=0; itest[2]<3; itest[2]++){
        for(int in=0; in<N3D; ++in){
            u[in][0] = GetCoor(in, 0)[itest[0]];
            u[in][1] = GetCoor(in, 1)[itest[1]];
            u[in][2] = GetCoor(in, 2)[itest[2]];
        }
        CalcFluxTerm_FD(u.data(), 1, 0, f.data()); // 1 - evaluate convection, 0 - do not evaluate viscosity
        int divu = (itest[0]==0)+(itest[1]==1)+(itest[2]==2); // velocity divergence (constant in space)
        real3 err;
        for(int in=0; in<N3D; ++in){
            int i[3] = {in%NX, (in/NX)%NY, in/(NX*NY)};
            const int BB = 6; // skip near-boundary strips of width BB (including periodic boundaries)
            if(i[0]<BB || i[0]>=NX-BB || i[1]<BB || i[1]>=NY-BB || i[2]<BB || i[2]>=NZ-BB) continue;
            real3 fex;
            real3 r = GetCoor(in, 0);
            fex[0] -= r[itest[0]]*divu + r[itest[itest[0]]];
            r = GetCoor(in, 1);
            fex[1] -= r[itest[1]]*divu + r[itest[itest[1]]];
            r = GetCoor(in, 2);
            fex[2] -= r[itest[2]]*divu + r[itest[itest[2]]];
            for(int icoor=0; icoor<3; icoor++){
                real errloc = fabs(f[in][icoor] - fex[icoor]);
                if(fabs(errloc) > 1e4){
                    errloc+=1.; // place for a breakpoint
                }
                err[icoor] = std::max(err[icoor], errloc);
            }
        }
        if(print_all) printf("Test %i %i %i: e = %.0e %.0e %.0e\n", itest[0], itest[1], itest[2], err[0], err[1], err[2]);
        max_err = std::max(max_err, abs(err));
    }
    printf("Explicit fluxes consistency: %.0e\n", max_err);
    if(print_all) printf("Explicit fluxes approximation check finished\n");
}


void data_ext::CheckGradientKernel(){
    #if(CONV_APPR_TYPE==CONV_VV)
    vector<real> p(N3D);
    for(int in=0; in<N3D; in++) p[in] = KERG[in/NXY];
    vector<real3> gradp(N3D);
    ApplyGradient(p.data(), gradp.data());
    real err = 0;
    for(int in=0; in<N3D; in++) err = std::max(abs(gradp[in]), err);
    printf("Gradient kernel check: %.0e\n", err);

    for(int in=0; in<N3D; in++) p[in] = sin(double(in)*in);
    NormalizePressure(p.data());
    vector<real> pcopy = p;
    NormalizePressure(p.data());
    err = 0;
    for(int in=0; in<N3D; in++) err = std::max(fabs(p[in] - pcopy[in]), err);
    printf("Is the pressure normalization actually a projection? = %.0e\n", err);
    #endif
}


void data_ext::CheckPressureMatrix(){
    // Matrix of the pressure solver (for the constant-in-xy mode)
    real MAT[7*NZ_MAX];
    FillMatrix(MAT);

    // Check symmetry
    real err_symm = 0;
    for(int i=0; i<NZ; i++){
        //printf("DIAG[%i]=%f\n", i, MAT[i*7+3]);
        for(int j=-3; j<=3; j++){
            if(i+j<0 || i+j>=NZ) continue;
            err_symm = std::max(err_symm, fabs(MAT[i*7+3+j] - MAT[(i+j)*7+3-j]));
        }
    }
    printf("Skew-symmetric part of the matrix: %.0e\n", err_symm);

    // Generate a pseudo-random pressure field, constant in XY
    vector<real> p(N3D);
    for(int iz=0; iz<NZ; iz++){
        p[iz*NXY] = sin(iz*iz+51.);
        for(int ixy=1; ixy<NXY; ixy++) p[iz*NXY+ixy]=p[iz*NXY];
    }

    // Evaluate div(grad(p))*diag{dz}
    vector<real3> gradp(N3D);
    vector<real> divgradp(N3D);
    ApplyGradient(p.data(), gradp.data());
    for(int i=0; i<N3D; i++) ApplyDivergence(i, gradp.data(), divgradp.data());

    // Same - using the matrix
    vector<real> divgradp_using_mat(NZ);
    for(int i=0; i<NZ; i++){
        divgradp_using_mat[i] = 0;
        for(int j=-3; j<=3; j++){
            if(i+j<0 || i+j>=NZ) continue;
            divgradp_using_mat[i] += MAT[7*i+j+3] * p[(i+j)*NXY];
        }
    }

    real err = 0.;
    for(int i=0; i<N3D; i++) err = std::max(err, fabs(divgradp_using_mat[i/NXY] - divgradp[i]));
    printf("Pressure matrix error: %.0e\n", err);
}

void data_ext::CheckPressureSolver(){
    for(int itest=0; itest<2; itest++){ // itest==0: pressure depends on 'z' only; itest==1: fully 3D
        // Generate a pseudo-random pressure field
        vector<real> p(N3D);
        for(int in=0; in<N3D; in++){
            int i[3] = {in%NX, (in/NX)%NY, in/(NX*NY)};
            if(itest==0) p[in] = sin(ZZ[i[2]]);//sin(i[2]);
            else p[in] = sin(double(in)*in);
        }
        NormalizePressure(p.data()); // to compare two pressure fields, both should have the same normalization

        // Evaluate div(grad(p))*diag{dz}
        vector<real3> gradp(N3D);
        vector<real> divgradp(N3D);
        ApplyGradient(p.data(), gradp.data());
        for(int i=0; i<N3D; i++) ApplyDivergence(i, gradp.data(), divgradp.data());

        // Try to get the original pressure field using the pressure solver
        vector<real> pnew(N3D);
        LinearSystemSolveFFT(pnew.data(), divgradp.data());
        NormalizePressure(pnew.data());

        // Evaluate div(grad(p))*diag{dz} for the new field
        vector<real3> gradp_new(N3D);
        vector<real> divgradp_new(N3D);
        ApplyGradient(pnew.data(), gradp_new.data());
        for(int i=0; i<N3D; i++) ApplyDivergence(i, gradp_new.data(), divgradp_new.data());

        double err = 0.;
        for(int in=0; in<N3D; in++){
            double errloc = fabs(p[in] - pnew[in]);
            err = std::max(err, errloc);
        }

        //for(int iz=0; iz<NZ; iz++) printf("%03i % .4f % .4f     % .4f % .4f\n", iz, p[iz*NXY], pnew[iz*NXY], divgradp[iz*NXY], divgradp_new[iz*NXY]);
        printf("Pressure solver error (test %i): %.0e\n", itest, err);
    }
}

// Momentum and kinetic energy preservation check
// itest==0: velocities depend on 'x' and 'y' only (one vortex mode, only on uniform mesh)
// itest==1: fully 3D velocity field
void data_ext::CheckConservation(int itest){
    #if(CONV_APPR_TYPE==CONV_VV)
    vector<real3> v(N3D), gradp(N3D), f(N3D);
    vector<real> p(N3D);

    if(itest==0){ // 2D field
        // Cannot just generate a random 2D field, because the projection onto the divergence-free space discards the 2D nature
        //for(int in=0; in<NXY; in++){
        //    for(int icoor=0; icoor<2; icoor++) v[in][icoor] = sin(double(in)*in+cos(double(10*icoor)));
        //    v[in][2]=0.;
        //}
        //for(int in=NXY; in<N3D; in++) v[in] = v[in%NXY];

        const int kx = 7, ky = 9; // wave numbers
        double Pi = 4.*atan(1.);
        double cx = sin(kx*Pi/NX) - sin(kx*3.*Pi/NX)/27.;
        double cy = sin(ky*Pi/NY) - sin(ky*3.*Pi/NY)/27.;
        for(int in=0; in<NXY; in++){
            int ix = in%NX, iy = in/NX;
            v[in][0] = (-cy)*cos(2.*Pi*(kx*(ix-0.5)/NX + ky*(iy-0.0)/NY));
            v[in][1] = ( cx)*cos(2.*Pi*(kx*(ix-0.0)/NX + ky*(iy-0.5)/NY));
            v[in][2] = 0;
        }
        for(int in=NXY; in<N3D; in++) v[in]=v[in%NXY];

        // Check that now div(u)=0
        UnapplyGradient(v.data(), p.data());
        real err_div = 0;
        for(int in=0; in<N3D; in++) err_div = std::max(err_div, fabs(p[in]));
        printf("Divergence norm of the generated velocity: %.0e\n", err_div);
    }
    else{ // 3D field
        for(int in=0; in<N3D; in++){
            //int i[3] = {in%NX, (in/NX)%NY, in/(NX*NY)};
            for(int icoor=0; icoor<3; icoor++) v[in][icoor] = sin(double(in)*in+cos(double(10*icoor)));
            if(in<NXY) v[in][2]=0.; // force w=0 on z=0 (these elements of the array do not correspond to actual DOFs)
        }

        // Enforce div(u) = 0
        UnapplyGradient(v.data(), p.data());
        ApplyGradient(p.data(), gradp.data());
        for(int in=0; in<N3D; in++) v[in] -= gradp[in];
        // Check that now div(u)=0
        UnapplyGradient(v.data(), p.data());
        real err_div = 0;
        for(int in=0; in<N3D; in++) err_div = std::max(err_div, fabs(p[in]));
        printf("Divergence norm of the projected velocity: %.0e\n", err_div);
    }

    CalcFluxTerm_FD(v.data(), 1, 0, f.data());

    double mx=0., my=0., mz=0., Kx=0., Ky=0., Kz=0.;
    for(int in=0; in<N3D; in++){
        int iz = in/NXY;
        double hz = 1.125*DZ[iz] - 0.125*DZW[iz];
        double hzw = 1./GCZ[iz];
        mx += f[in][0] * hz;
        my += f[in][1] * hz;
        if(in/NXY>0) mz += f[in][2] * hzw;
        Kx += f[in][0]*v[in][0] * hz;
        Ky += f[in][1]*v[in][1] * hz;
        if(in/NXY>0) Kz += f[in][2]*v[in][2] * hzw;
    }
    // mz should be equal to boundary forces (of unknown value), so do not print it
    // for the 3D field with boundaries, mx and my momemtum is nonzero as well
    if(itest) printf("Conservation test (3D): K=%.0e %.0e %.0e\n", Kx, Ky, Kz);
    else printf("Conservation test (2D): mx=%.0e my=%.0e K=%.0e %.0e %.0e\n", mx, my, Kx, Ky, Kz);
    #endif
}

void data_ext::CheckViscNaiveConsistency(int IsMeshUniform, int print_all){
    // Set turbulent viscosity to zero (if the array is not allocated, then it is zero automatically)
    if(nu_t!=NULL) for(int in=0; in<N3D; in++) nu_t[in]=0;
    vector<real3> v(N3D), f(N3D);

    // Check consistency first. By linearity, it is enough to check the one-component velocity fields
    if(print_all) printf("Naive viscous fluxes approximation check:\n");
    real max_err = 0;
    for(int icomponent=0; icomponent<3; icomponent++){
        // Loop in polynomials, of degree at most two in each coordinate
        //int px=1, py=1, pz=0;{
        for(int px=0; px<=2; px++) for(int py=0; py<=2; py++) for(int pz=0; pz<=2; pz++){
            if(!IsMeshUniform && pz==2) continue; // on a non-uniform mesh, there is no exactness in quadratic polynomials in Z
            for(int in=0; in<N3D; in++){
                real3 r = GetCoor(in, icomponent);
                double vel = 1;
                for(int i=0; i<px; i++) vel*=r[0];
                for(int i=0; i<py; i++) vel*=r[1];
                for(int i=0; i<pz; i++) vel*=r[2];
                v[in] = real3();
                v[in][icomponent] = vel;
                if(in<NXY) v[in][2]=0; // does not matter, as we do not look at fluxes near boundaries
            }

            //Calculate fluxes here - of after
            //CalcFluxTerm_FD(v.data(), 0, 1, f.data());

            real3 err;
            for(int in=0; in<N3D; ++in){
                int i[3] = {in%NX, (in/NX)%NY, in/(NX*NY)};
                const int BB = 3; // skip near-boundary strips of width BB (including periodic boundaries)
                if(i[0]<BB || i[0]>=NX-BB || i[1]<BB || i[1]>=NY-BB || i[2]<BB || i[2]>=NZ-BB) continue;
                real3 fex;

                // Calculate fluxes here - if not calculated before
                f[in]=ViscFlux<tViscApproximation::NAIVE>(in,v.data());

                // renormalize (no effect on uniform meshes)
                /*{
                    int iz = i[2];
                    f[in][0] *= 0.125*(9*DZ[iz] - DZW[iz]) / (0.625*DZ[iz] + 0.375*DZW[iz]);
                    f[in][1] *= 0.125*(9*DZ[iz] - DZW[iz]) / (0.625*DZ[iz] + 0.375*DZW[iz]);
                }*/
                f[in] /= visc;

                // Grad(Div(u))
                for(int ivar=0; ivar<3; ivar++){
                    int qx = px - (icomponent==0) - (ivar==0);
                    int qy = py - (icomponent==1) - (ivar==1);
                    int qz = pz - (icomponent==2) - (ivar==2);
                    if(qx<0 || qy<0 || qz<0) continue;

                    real3 r = GetCoor(in, ivar);
                    double vel = pow(r[0], qx) * pow(r[1], qy) * pow(r[2], qz);
                    if(qx<px && px==2) vel*=2;
                    if(qy<py && py==2) vel*=2;
                    if(qz<pz && pz==2) vel*=2;

                    fex[ivar] += vel;
                }
                // Laplace(u) (only for ivar==icomponent)
                {
                    real3 r = GetCoor(in, icomponent);
                    if(px==2) fex[icomponent] += 2 * pow(r[1], py) * pow(r[2], pz);
                    if(py==2) fex[icomponent] += 2 * pow(r[0], px) * pow(r[2], pz);
                    if(pz==2) fex[icomponent] += 2 * pow(r[0], px) * pow(r[1], py);
                }

                for(int icoor=0; icoor<3; icoor++){
                    real errloc = fabs(f[in][icoor] - fex[icoor]);
                    if(errloc > 1e-5){
                        errloc = errloc;
                    }
                    err[icoor] = std::max(err[icoor], errloc);
                }
                //printf("%i %06i %e %e %e\n", icomponent, in/NXY, f[in][1]/visc, fex[1]/visc, f[in][1]/fex[1]);
            }

            if(print_all) printf("Test %c %i %i %i: e = %.0e %.0e %.0e\n", 'u'+icomponent, px, py, pz, err[0], err[1], err[2]);
            max_err = std::max(max_err, abs(err));
        }
    }
    printf("Naive viscous fluxes consistency: %.0e\n", max_err);
    if(print_all) printf("Naive viscous fluxes approximation check finished\n");
}



// Subroutines check (CPU only)
void data_ext::MakeCPUChecks(int IsMeshUniform){
    #if(CONV_APPR_TYPE==CONV_VV)
    if(IsMeshUniform) CheckConvectionConsistency(false);
    CheckGradientKernel();
    CheckPressureMatrix();
    CheckPressureSolver();
    if(IsMeshUniform) CheckConservation(0);
    CheckConservation(1);
    CheckViscNaiveConsistency(IsMeshUniform, false);
    //CheckViscGalerkinConsistency(IsMeshUniform, true);
    #endif
}


// Check that CPU and GPU give the same results
void data_ext::CheckGPU_ApplyGradient(data_cuda::data& G){
    for(int in=0; in<N3D; in++) pres[0][in] = sin(double(in)*in);
    GPU_CHECK_CRASH( cudaMemcpy( G.pres[0], pres[0], N3D*sizeof(real), cudaMemcpyHostToDevice ) );
    G.ApplyGradient(G.pres[0], G.vel[0]);
    ApplyGradient(pres[0], vel[0]);
    GPU_CHECK_CRASH( cudaMemcpy( vel[1], G.vel[0], N3D*sizeof(real3), cudaMemcpyDeviceToHost ) );
    GPU_CHECK_CRASH( cudaDeviceSynchronize() );
    real err = 0.;
    for(int in=0; in<N3D; in++) err = std::max(err, abs(vel[0][in]-vel[1][in]));
    printf("CPU-GPU check: ApplyGradient: %.0e\n", err);
}

void data_ext::CheckGPU_UnapplyGradientAndExplicit(data_cuda::data& G){
    for(int in=0; in<N3D; ++in){
        int i[3] = {in%NX, (in/NX)%NY, in/(NX*NY)};
        double x = i[0]*hx, /*y = i[1]*hy,*/ z = ZZ[i[2]];
        double H = 0.5*(ZZ[NZ] - ZZ[0]), zmin = ZZ[0];
        double u_tau = 1., Hz = 2.*H;
        double zplus = std::min(z, 2.*H-z)*u_tau/visc;
        vel[0][in][0] = 0.5*(z-zmin)*(zmin+Hz-z)*sour_dpdx/visc;
        vel[0][in][1] = 1e-1*sin(0.5*x)*zplus*exp(-0.01*zplus*zplus);
        vel[0][in][2] = 0.;
        if(nu_t!=NULL) nu_t[in] = 0.;
        for(int idir=0; idir<3; idir++) if(i[2]!=0 && i[2]!=NZ-1) vel[0][in][idir] += 1e-1*sin(double(in)*in);
    }
    if(nu_t!=NULL) GPU_CHECK_CRASH( cudaMemcpy( G.nu_t, nu_t, N3D*sizeof(real), cudaMemcpyHostToDevice ) );
    GPU_CHECK_CRASH( cudaMemcpy( G.vel[0], vel[0], N3D*sizeof(real3), cudaMemcpyHostToDevice ) );
    G.UnapplyGradient(G.vel[0], G.pres[0]);
    UnapplyGradient(vel[0], pres[0]);
    GPU_CHECK_CRASH( cudaMemcpy( pres[1], G.pres[0], N3D*sizeof(real), cudaMemcpyDeviceToHost ) );
    cudaDeviceSynchronize();
    real err = 0.;
    for(int in=0; in<N3D; in++) err += fabs(pres[0][in]-pres[1][in]);
    printf("CPU-GPU check: UnapplyGradient: %.0e\n", err);

    G.ExplicitTerm(G.vel[0], G.vel[1]);
    ExplicitTerm(vel[0], vel[1]);
    GPU_CHECK_CRASH( cudaMemcpy( vel[2], G.vel[1], N3D*sizeof(real3), cudaMemcpyDeviceToHost ) );
    cudaDeviceSynchronize();
    err = 0.;
    for(int in=0; in<N3D; in++) err = std::max(err, abs(vel[1][in]-vel[2][in]));
    printf("CPU-GPU check: ExplicitTerm: %.0e\n", err);
}

void data_ext::CheckGPU_CalcProfiles(data_cuda::data& G){
    thrust::host_vector<real3> v(N3D);
    thrust::host_vector<real> p(N3D);
    for(int in=0; in<N3D; in++){
        for(int icoor=0; icoor<3; icoor++) v[in][icoor] = sin(real(in*in+icoor*10+5));
        if(in<NXY) v[in][2]=0;
        p[in] = sin(real(in*in+87));
    }
    CalcProfiles(v.data(), p.data());
    vector< real9<real> > av_host(NZ);
    for(int iz=0; iz<NZ; iz++) av_host[iz]=av_u[iz];

    thrust::device_vector<real3> dev_v = v;
    thrust::device_vector<real> dev_p = p;
    G.CalcProfiles(thrust::raw_pointer_cast(dev_v.data()), thrust::raw_pointer_cast(dev_p.data()));
    GPU_CHECK_CRASH( cudaMemcpy( av_u, G.av_u, NZ*sizeof(real9<real>), cudaMemcpyDeviceToHost ) );

    real err = 0.;
    for(int iz=0; iz<NZ; iz++){
        real errloc = 0;
        for(int i=0; i<3; i++) errloc += fabs(av_host[iz].u[i] - av_u[iz].u[i]);
        for(int i=0; i<6; i++) errloc += fabs(av_host[iz].uu[i] - av_u[iz].uu[i]);
        errloc += fabs(av_host[iz].p - av_u[iz].p);
        errloc += fabs(av_host[iz].pp - av_u[iz].pp);
        err = std::max(err, errloc);
    }
    printf("CPU-GPU check: CalcProfiles: %.0e\n", err);
}

// Comparison of specific subroutines on CPU and GPU
void data_ext::MakeCPUGPUChecks(data_cuda::data& DATA_CUDA){
    CheckGPU_ApplyGradient(DATA_CUDA);
    CheckGPU_UnapplyGradientAndExplicit(DATA_CUDA);
    CheckGPU_CalcProfiles(DATA_CUDA);
}
