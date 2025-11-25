// Fluxes (convective and diffusive, combined for the sake of performance)

// Version with o{x,y,z} = {0,1}
__DEVICE__ inline int GetElementIndex(int in, int ox, int oy, int oz){
    if(!ox) in = XL(in);
    if(!oy) in = YL(in);
    if(!oz) in = ZL(in);
    return in;
}

// Version with o{x,y,z} = {-1,1}
__DEVICE__ inline int GetElementIndexA(int in, int ox, int oy, int oz){
    if(ox!=1) in = XL(in);
    if(oy!=1) in = YL(in);
    if(oz!=1) in = ZL(in);
    return in;
}


// Naive implementation of the viscous term
__DEVICE__ real3 data::CalcViscTermGalerkin(int in, const real3* u) const{
    real3 sumflux;

    for(int oz=0; oz<=1; oz++){
        if(FDmode){
            if(IsWall(in)) continue;
        }
        else{
            if((oz==0 && in<NX*NY) || (oz==1 && in>=(NZ-1)*NX*NY)) continue;
        }
        int in_z = oz ? ZR(in) : ZL(in);
        int iz = in/(NX*NY);
        real hz = oz==0 ? DZ[iz==0?0:iz-1] : DZ[iz];
        real inv_hz = 1./hz;

        for(int oy=0; oy<=1; oy++){
            int in_y = oy ? YR(in) : YL(in);
            int in_yz = oy ? YR(in_z) : YL(in_z);

            for(int ox=0; ox<=1; ox++){
                int in_x   = ox ? XR(in) : XL(in);
                int in_xy  = ox ? XR(in_y) : XL(in_y);
                int in_xz  = ox ? XR(in_z) : XL(in_z);
                int in_xyz = ox ? XR(in_yz) : XL(in_yz);

                real mx = inv_hx*hy*hz;
                real my = inv_hy*hx*hz;
                real mz = inv_hz*hx*hy;

                // nabla_j * (nu nabla_j u_i)
                real3 g = (C1_9*(u[in_x]-u[in]) + C1_18*(u[in_xy]-u[in_y]) + C1_18*(u[in_xz]-u[in_z]) + C1_36*(u[in_xyz]-u[in_yz]))*mx +
                          (C1_9*(u[in_y]-u[in]) + C1_18*(u[in_xy]-u[in_x]) + C1_18*(u[in_yz]-u[in_z]) + C1_36*(u[in_xyz]-u[in_xz]))*my +
                          (C1_9*(u[in_z]-u[in]) + C1_18*(u[in_xz]-u[in_x]) + C1_18*(u[in_yz]-u[in_y]) + C1_36*(u[in_xyz]-u[in_xy]))*mz;

                // nabla_j * (nu nabla_i u_j)
                real gxx = C1_9*(u[in_x][0]-u[in][0]) + C1_18*(u[in_xy][0]-u[in_y][0]) + C1_18*(u[in_xz][0]-u[in_z][0]) + C1_36*(u[in_xyz][0]-u[in_yz][0]);
                real gxy = real(0.25)*(C1_3*(u[in_x][1]-u[in][1] + u[in_xy][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_z][1] + u[in_xyz][1]-u[in_yz][1]));
                real gxz = real(0.25)*(C1_3*(u[in_x][2]-u[in][2] + u[in_xz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_y][2] + u[in_xyz][2]-u[in_yz][2]));
                real gyy = C1_9*(u[in_y][1]-u[in][1]) + C1_18*(u[in_xy][1]-u[in_x][1]) + C1_18*(u[in_yz][1]-u[in_z][1]) + C1_36*(u[in_xyz][1]-u[in_xz][1]);
                real gyx = real(0.25)*(C1_3*(u[in_y][0]-u[in][0] + u[in_xy][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_z][0] + u[in_xyz][0]-u[in_xz][0]));
                real gyz = real(0.25)*(C1_3*(u[in_y][2]-u[in][2] + u[in_yz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_x][2] + u[in_xyz][2]-u[in_xz][2]));
                real gzz = C1_9*(u[in_z][2]-u[in][2]) + C1_18*(u[in_yz][2]-u[in_y][2]) + C1_18*(u[in_xz][2]-u[in_x][2]) + C1_36*(u[in_xyz][2]-u[in_xy][2]);
                real gzx = real(0.25)*(C1_3*(u[in_z][0]-u[in][0] + u[in_xz][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_y][0] + u[in_xyz][0]-u[in_xy][0]));
                real gzy = real(0.25)*(C1_3*(u[in_z][1]-u[in][1] + u[in_yz][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_x][1] + u[in_xyz][1]-u[in_xy][1]));

                gxx*=mx;
                gxy*=hz*(ox^oy ? -1:1);
                gxz*=hy*(ox^oz ? -1:1);
                gyy*=my;
                gyx*=hz*(ox^oy ? -1:1);
                gyz*=hx*(oy^oz ? -1:1);
                gzz*=mz;
                gzx*=hy*(ox^oz ? -1:1);
                gzy*=hx*(oy^oz ? -1:1);

                g += real3(gxx+gxy+gxz, gyy+gyx+gyz, gzz+gzx+gzy);

                real visc_coeff = visc;
                if(nu_t!=NULL) visc_coeff += nu_t[GetElementIndex(in, ox, oy, oz)];
                sumflux += g*visc_coeff;
            }
        }
    }
    return sumflux;
}

__DEVICE__ void data::CalcFluxTerm_FD(int in, const real3* u, int DoConv, int DoVisc, real3* kuhat) const{
    real3 sumflux;
    if(!IsWall(in)){
        int iz = in/(NX*NY);
        real inv_hz = 2/(DZ[iz-1]+DZ[iz]);
        if(DoConv){ // Convection
            const real3 inv_MeshStep(inv_hx, inv_hy, inv_hz);

            for(int idir=0; idir<3; idir++){
                const double invh = inv_MeshStep[idir];
                int jn = Neighb(in, idir, 0);
                int kn = Neighb(in, idir, 1);

                real3 conv_div = real(0.5)*(u[kn]*u[kn][idir] - u[jn]*u[jn][idir]);
                real3 conv_adv = u[in][idir] * real(0.5)*(u[kn]-u[jn]);
                sumflux -= real(0.5)*(conv_div + conv_adv) * invh;
            }
        }

        // Viscosity
        if(DoVisc){
            #if VISC_TERM_GALERKIN==1
                sumflux+=CalcViscTermGalerkin(in, u) * inv_hx*inv_hy*inv_hz;
            #else
                sumflux += visc*inv_hx*inv_hx*(u[XR(in)]+u[XL(in)]-2.*u[in]);
                sumflux += visc*inv_hy*inv_hy*(u[YR(in)]+u[YL(in)]-2.*u[in]);
                sumflux += visc*((u[ZR(in)]-u[in])/DZ[iz] - (u[in]-u[ZL(in)])/DZ[iz-1])*inv_hz;
            #endif
        }
        // Source term
        sumflux[0] += sour_dpdx;
    }
    kuhat[in] = sumflux;
}

void data::CalcFluxTerm_FD(double t, const real3* u, const real* p, int DoConv, int DoVisc, real3* kuhat) const{
    Forall([u,DoConv,DoVisc,kuhat] __DEVICE__ (int i){ D.CalcFluxTerm_FD(i,u,DoConv,DoVisc,kuhat); });
}

// For CPU code, use a naive implementation
// If the viscosity is approximated by the `direct cross` scheme (only for constant viscosity),
// then use a naive parallel implementation for GPU as well
#if VISC_TERM_GALERKIN==0 || !defined(THIS_IS_CUDA)

void data::ExplicitTerm(double t, const real3* u, const real* p, real3* kuhat) const{
    t_ExplicitTerm.beg();
    Forall([t,u,p,kuhat] __DEVICE__ (int i){ D.CalcFluxTerm_FD(i,u,true,true,kuhat); });
    t_ExplicitTerm.end();
}

#else

// Special implementation for CUDA for VISC_TERM_GALERKIN==1
#define BLK_SHM_X 8
#define BLK_SHM_Y 8
#define BLK_SHM_Z 8
#define BLK_SHM_THR_Z 4 // actual number threads in block (if BLK_SHM_THR_Z<BLK_SHM_Z, then each thread will process several nodes)
#define SHY (BLK_SHM_X+2)
#define SHZ (BLK_SHM_Y+2)*(BLK_SHM_X+2)

__global__ void CalcFluxTerm_FD_ShM(const real3* uu, real3* kuhat){
    __shared__ _real3 u[(BLK_SHM_Z+2)*(BLK_SHM_Y+2)*(BLK_SHM_X+2)];

    // Reading data to shared memory
    for(int i=threadIdx.x+BLK_SHM_X*(threadIdx.y+BLK_SHM_Y*threadIdx.z); i<(BLK_SHM_Z+2)*(BLK_SHM_Y+2)*(BLK_SHM_X+2); i+=BLK_SHM_X*BLK_SHM_Y*BLK_SHM_THR_Z){
        int ix = (i%(BLK_SHM_X+2)                 + blockIdx.x*BLK_SHM_X + (NX-1)) % NX;
        int iy = ((i/(BLK_SHM_X+2))%(BLK_SHM_Y+2) + blockIdx.y*BLK_SHM_Y + (NY-1)) % NY;
        int iz = i/((BLK_SHM_X+2)*(BLK_SHM_Y+2))  + blockIdx.z*BLK_SHM_Z - 1;
        int in = ix + iy*NX + iz*(NX*NY);

        if(iz>=0 && iz<D.NZ && i<(BLK_SHM_Z+2)*(BLK_SHM_Y+2)*(BLK_SHM_X+2)) u[i] = uu[in];
    }

    // Ensure that all threads wrote the data to the shared memory
    __syncthreads();

    for(int threadIdx_z=threadIdx.z; threadIdx_z<BLK_SHM_Z; threadIdx_z+=BLK_SHM_THR_Z){
        real3 sumflux;
        // Mesh index of the node, owned by the thread
        int In = (threadIdx.x + blockIdx.x*BLK_SHM_X) + (threadIdx.y + blockIdx.y*BLK_SHM_Y)*NX + (threadIdx_z + blockIdx.z*BLK_SHM_Z)*NX*NY;
        // Index of the node in the shared fragment
        int i = threadIdx.x+1 + SHY*(threadIdx.y+1) + SHZ*(threadIdx_z+1);

        if(In<D.N3D && !D.IsWall(In)){
            int iz = In/(NX*NY);
            real inv_hbarz = real(2)/(DZ[iz-1]+DZ[iz]);

            { // convection
                {
                    real3 conv_div = u[i+1]*u[i+1][0] - u[i-1]*u[i-1][0];
                    real3 conv_adv = u[i][0] * (u[i+1]-u[i-1]);
                    sumflux -= real(0.25)*(conv_div + conv_adv) * D.inv_hx;
                }
                {
                    real3 conv_div = u[i+SHY]*u[i+SHY][1] - u[i-SHY]*u[i-SHY][1];
                    real3 conv_adv = u[i][1] * (u[i+SHY]-u[i-SHY]);
                    sumflux -= real(0.25)*(conv_div + conv_adv) * D.inv_hy;
                }
                {
                    real3 conv_div = u[i+SHZ]*u[i+SHZ][2] - u[i-SHZ]*u[i-SHZ][2];
                    real3 conv_adv = u[i][2] * (u[i+SHZ]-u[i-SHZ]);
                    sumflux -= real(0.25)*(conv_div + conv_adv) * inv_hbarz;
                }
            }

            // Viscosity
            const int in = i;
            for(int oz=-1; oz<=1; oz+=2){
                if(D.FDmode){
                    if(D.IsWall(In)) continue;
                }
                else{
                    if((oz==-1 && In<NX*NY) || (oz==1 && In>=(D.NZ-1)*NX*NY)) continue;
                }
                int in_z = in+SHZ*oz;
                real hz = oz==-1 ? DZ[iz==0?0:iz-1] : DZ[iz];
                real inv_hz = 1./hz;

                for(int oy=-1; oy<=1; oy+=2){
                    int in_y = in+oy*SHY;
                    int in_yz = in_z+oy*SHY;

                    for(int ox=-1; ox<=1; ox+=2){
                        int in_x   = in+ox;
                        int in_xy  = in_y+ox;
                        int in_xz  = in_z+ox;
                        int in_xyz = in_yz+ox;

                        real visc_coeff = D.visc;
                        if(D.nu_t!=NULL) visc_coeff += D.nu_t[GetElementIndexA(In, ox, oy, oz)];
                        visc_coeff *= D.inv_hx*D.inv_hy*inv_hbarz;

                        real mx = D.inv_hx*D.hy*hz;
                        real my = D.inv_hy*D.hx*hz;
                        real mz = inv_hz*D.hx*D.hy;

                        // nabla_j * (nu nabla_j u_i)
                        {
                        real3 g = (C1_9*(u[in_x]-u[in]) + C1_18*(u[in_xy]-u[in_y]) + C1_18*(u[in_xz]-u[in_z]) + C1_36*(u[in_xyz]-u[in_yz]))*mx +
                                  (C1_9*(u[in_y]-u[in]) + C1_18*(u[in_xy]-u[in_x]) + C1_18*(u[in_yz]-u[in_z]) + C1_36*(u[in_xyz]-u[in_xz]))*my +
                                  (C1_9*(u[in_z]-u[in]) + C1_18*(u[in_xz]-u[in_x]) + C1_18*(u[in_yz]-u[in_y]) + C1_36*(u[in_xyz]-u[in_xy]))*mz;
                        sumflux += g*visc_coeff;
                        }

                        // nabla_j * (nu nabla_i u_j)
                        // gxx
                        real g = C1_9*(u[in_x][0]-u[in][0]) + C1_18*(u[in_xy][0]-u[in_y][0]) + C1_18*(u[in_xz][0]-u[in_z][0]) + C1_36*(u[in_xyz][0]-u[in_yz][0]);
                        sumflux[0] += g*mx*visc_coeff;
                        // gxy
                        g = real(0.25)*(C1_3*(u[in_x][1]-u[in][1] + u[in_xy][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_z][1] + u[in_xyz][1]-u[in_yz][1]));
                        sumflux[0] += g*hz*(ox*oy)*visc_coeff;
                        // gxz
                        g = real(0.25)*(C1_3*(u[in_x][2]-u[in][2] + u[in_xz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_y][2] + u[in_xyz][2]-u[in_yz][2]));
                        sumflux[0] += g*D.hy*(ox*oz)*visc_coeff;
                        // gyy
                        g = C1_9*(u[in_y][1]-u[in][1]) + C1_18*(u[in_xy][1]-u[in_x][1]) + C1_18*(u[in_yz][1]-u[in_z][1]) + C1_36*(u[in_xyz][1]-u[in_xz][1]);
                        sumflux[1] += g*my*visc_coeff;
                        // gyx
                        g = real(0.25)*(C1_3*(u[in_y][0]-u[in][0] + u[in_xy][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_z][0] + u[in_xyz][0]-u[in_xz][0]));
                        sumflux[1] += g*hz*(ox*oy)*visc_coeff;
                        // gyz
                        g = real(0.25)*(C1_3*(u[in_y][2]-u[in][2] + u[in_yz][2]-u[in_z][2]) + C1_6*(u[in_xy][2]-u[in_x][2] + u[in_xyz][2]-u[in_xz][2]));
                        sumflux[1] += g*D.hx*(oy*oz)*visc_coeff;
                        // gzz
                        g = C1_9*(u[in_z][2]-u[in][2]) + C1_18*(u[in_yz][2]-u[in_y][2]) + C1_18*(u[in_xz][2]-u[in_x][2]) + C1_36*(u[in_xyz][2]-u[in_xy][2]);
                        sumflux[2] += g*mz*visc_coeff;
                        // gzx
                        g = real(0.25)*(C1_3*(u[in_z][0]-u[in][0] + u[in_xz][0]-u[in_x][0]) + C1_6*(u[in_yz][0]-u[in_y][0] + u[in_xyz][0]-u[in_xy][0]));
                        sumflux[2] += g*D.hy*(ox*oz)*visc_coeff;
                        // gzy
                        g = real(0.25)*(C1_3*(u[in_z][1]-u[in][1] + u[in_yz][1]-u[in_y][1]) + C1_6*(u[in_xz][1]-u[in_x][1] + u[in_xyz][1]-u[in_xy][1]));
                        sumflux[2] += g*D.hx*(oy*oz)*visc_coeff;
                    }
                }
            }
            // Source term
            sumflux[0] += D.sour_dpdx;
        }
        if(In<D.N3D) kuhat[In] = sumflux;
    }
}

void data::ExplicitTerm(double t, const real3* u, const real* p, real3* kuhat) const{
    t_ExplicitTerm.beg();

    dim3 threads(BLK_SHM_X, BLK_SHM_Y, BLK_SHM_THR_Z);
    dim3 blocks(NX/BLK_SHM_X, NY/BLK_SHM_Y, (NZ+BLK_SHM_Z-1)/BLK_SHM_Z);
    CalcFluxTerm_FD_ShM<<<blocks,threads>>>(u, kuhat);
    GPU_CHECK_CRASH(cudaGetLastError());
    #ifdef SYNC_AFTER_EACH_KERNEL
        GPU_CHECK_CRASH(cudaDeviceSynchronize());
    #endif

    t_ExplicitTerm.end();
}
#endif // Special implementation
