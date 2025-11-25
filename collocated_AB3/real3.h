#ifndef REAL3_H
#define REAL3_H

// Vector with 3 components
// Version with no explicit constructor (to use in constant/shared memory)
struct _real3{
    real V[3];

    __device__ __host__ inline operator       real*()       {return V;}
    __device__ __host__ inline operator const real*() const {return V;}

    __device__ __host__ inline       real& operator[](int i){ return V[i];}
    __device__ __host__ inline const real& operator[](int i)const{ return V[i];}
    __device__ __host__ inline _real3& operator*=(const real &x){ V[0]*=x; V[1]*=x; V[2]*=x; return *this; }
    __device__ __host__ inline _real3& operator/=(const real &x){ real inv_x = 1./x; V[0]*=inv_x; V[1]*=inv_x; V[2]*=inv_x; return *this; }
    __device__ __host__ inline _real3& operator+=(const _real3& o){ V[0]+=o.V[0]; V[1]+=o.V[1]; V[2]+=o.V[2]; return *this; }
    __device__ __host__ inline _real3& operator-=(const _real3& o){ V[0]-=o.V[0]; V[1]-=o.V[1]; V[2]-=o.V[2]; return *this; }
};

__device__ __host__ inline _real3 operator+(const _real3 &a, const _real3 &b){_real3 R(a); return R += b;}
__device__ __host__ inline _real3 operator-(const _real3 &a, const _real3 &b){_real3 R(a); return R -= b;}
__device__ __host__ inline _real3 operator*(const _real3 &a, const real &b){_real3 R(a); return R *= b;}
__device__ __host__ inline _real3 operator/(const _real3 &a, const real &b){_real3 R(a); return R /= b;}
__device__ __host__ inline _real3 operator*(const real &a, const _real3 &b){_real3 R(b); return R *= a;}

__device__ __host__ inline real DotProd(const _real3 &a, const _real3 &b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
__device__ __host__ inline real abs(const _real3 &a) { return sqrt(DotProd(a,a)); }

// Main version
struct real3 : _real3{
    __device__ __host__ inline real3(){ V[0]=V[1]=V[2]=0.; }
    __device__ __host__ inline real3(const real3& b){ V[0]=b.V[0]; V[1]=b.V[1]; V[2]=b.V[2]; }
    __device__ __host__ inline real3(const _real3& b){ V[0]=b.V[0]; V[1]=b.V[1]; V[2]=b.V[2]; }
    __device__ __host__ inline real3(real x, real y, real z){ V[0]=x; V[1]=y; V[2]=z; }
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


#endif

