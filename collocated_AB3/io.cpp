#include <stdio.h>
#include <fstream>
#include "CLI11.hpp"
#include "config.h"
#include "real3.h"
#include "io.h"


void DumpData(const main_params& MD, const double* Z, const char* fname, const real3* velocity, const real* pressure, const real* qressure){
    const int N[3] = {NX, NY, MD.NZ};
    const int NN = N[0]*N[1]*N[2];
    FILE* f = fopen(fname, "wt");
    fprintf(f, "# vtk DataFile Version 2.0\n");
    fprintf(f, "Volume example\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET RECTILINEAR_GRID\n");
    fprintf(f, "DIMENSIONS %i %i %i\n", N[0], N[1], N[2]);
    for(int idir=0; idir<3; idir++){
        fprintf(f, "%c_COORDINATES %i float\n", 'X'+idir, N[idir]);
        for(int i=0; i<N[idir]; i++) fprintf(f, "%e ", (idir==0) ? MD.xmin+MD.hx*i : (idir==1) ? MD.ymin+MD.hy*i : Z[i]);
        fprintf(f, "\n");
    }
    fprintf(f, "POINT_DATA %i\n", NN);
    fprintf(f, "SCALARS volume_scalars float %i\n", 4+(qressure!=NULL));
    fprintf(f, "LOOKUP_TABLE default\n");
    for(int in=0; in<NN; ++in){
        fprintf(f, "%e %e %e %e", velocity[in][0], velocity[in][1], velocity[in][2], pressure[in]);
        if(qressure!=NULL) fprintf(f, " %e", qressure[in]);
        fprintf(f, "\n");
    }
    fclose(f);
}

int ReadData(const main_params& MD, const char* fname, real3* velocity, real* pressure){
    const int N[3] = {NX, NY, MD.NZ};
    std::ifstream file(fname);
    if(!file.is_open()) return 1;
    std::string line;
    int iz=-1, n[3]={-1,-1,-1};
    while(std::getline(file, line)) {
        if(line.substr(0, 10)==std::string("DIMENSIONS")){
            line = line.substr(11);
            sscanf(line.c_str(), "%i%i%i", n, n+1, n+2);
            if(n[2]!=N[2]){ printf("NZ mismatch\n"); exit(0); }
            if(N[0]%n[0] || N[1]%n[1]) { printf("NX or NY is wrong\n"); exit(0); }
        }
        if(line.substr(0, 10)==std::string("LOOKUP_TAB")) {iz=0; break;}
    }
    if(iz==-1 || n[0]<0 || n[1]<0 || n[2]<0) { printf("Unexpected EOF (no data read)\n"); exit(0); }

    // Mode for duplicating the data (h changed, box size was not => repeat each value)
    printf("Stored data %ix%ix%i, reading to %ix%ix%i\n", n[0], n[1], n[2], N[0], N[1], N[2]);
    int MX = N[0]/n[0], MY = N[1]/n[1];
    for(iz=0; iz<n[2]; iz++){
        for(int iy=0; iy<n[1]; iy++){
            for(int ix=0; ix<n[0]; ix++){
                if(!std::getline(file, line)) { printf("Unexpected EOF\n"); exit(0); }
                int in=iz*N[1]*N[0] + iy*N[0]*MY + ix*MX;
                sscanf(line.c_str(), sizeof(real)==8 ? "%lf%lf%lf%lf" : "%f%f%f%f", &(velocity[in][0]), &(velocity[in][1]), &(velocity[in][2]), &(pressure[in]));
                for(int iiy=0; iiy<MY; iiy++) for(int iix=0; iix<MX; iix++){
                    if(iiy==0 && iix==0) continue;
                    int jn = in + iiy*N[0] + iix;
                    velocity[jn] = velocity[in];
                    pressure[jn] = pressure[in];
                }
            }
        }
    }
    file.close();
    return 0;
}


void DumpProfiles(const main_params& MD, const double* Z, const real9* av, double visc, double u_fric, const char* fname, int IsZplus){
    FILE* F = fopen(fname, "wt");

    for(int iz=0; iz<MD.NZ; iz++){
        double H = 0.5*(Z[MD.NZ-1] - Z[0]);
        double dist_to_wall = (Z[iz]<H) ? Z[iz] : 2.*H-Z[iz];
        double y_plus = dist_to_wall*u_fric;
        if(!IsZplus) y_plus /= visc;

        real3 u = av[iz].u / u_fric;
        double uu[6];
        for(int i=0; i<6; i++) uu[i] = av[iz].uu[i] / (u_fric*u_fric);
        uu[0] = std::max(0., uu[0]-u[0]*u[0]);
        uu[1] = std::max(0., uu[1]-u[1]*u[1]);
        uu[2] = std::max(0., uu[2]-u[2]*u[2]);
        uu[3] -= u[0]*u[1];
        uu[4] -= u[0]*u[2];
        uu[5] -= u[1]*u[2];

        double nu_t; // turbulent viscosity is defined at half-integer points, need to average
        if(iz==0) nu_t = av[0].nu_t;
        else if(iz==MD.NZ-1) nu_t = av[iz-1].nu_t;
        else nu_t = 0.5*(av[iz].nu_t + av[iz-1].nu_t);

        fprintf(F, "%e %e %e %e %e %e %e\n", y_plus, u[0], sqrt(uu[0]), sqrt(uu[1]), sqrt(uu[2]), uu[4], nu_t/visc);
    }
    fclose(F);
}


static int _ReadCmdlineParams(int argc, char** argv, int DeviceCount, t_cmdline_params& PM){
    CLI::App app{"Turbulent flow between parallel plates"};
    argv = app.ensure_utf8(argv);

    std::string filename = "default";
    app.add_option("-d,--device", PM.DeviceID, std::string("device to use, 0..") + std::to_string(DeviceCount-1));
    app.add_option("-t,--TimeMax", PM.TimeMax, "maximal integration time");
    app.add_option("-a,--TimeStartAveraging", PM.TimeStartAveraging, "time where averaging should be started");
    app.add_option("--TimeStepsMax", PM.TimeStepsMax, "maximal number of iterations");
    app.add_option("--tau", PM.tau, "initial timestep size");
    app.add_option("--CFLmin", PM.CFLmin, "if CFL falls below this value, then the timestep will be increased");
    app.add_option("--CFLmax", PM.CFLmax, "if CFL lifts above this value, then the timestep will be decreased");
    app.add_option("-s,--SProgress", PM.SProgress, "number of timesteps between writing to screen and log");
    app.add_option("-f,--FProgress", PM.FProgress, "number of timesteps between writing a field");
    app.add_option("-p,--PProgress", PM.PProgress, "number of timesteps between writing a profile");

    CLI11_PARSE(app, argc, argv); // returns 0 if --help set
    return 1;
}

int ReadCmdlineParams(int argc, char** argv, int DeviceCount, t_cmdline_params& PM){
    return !_ReadCmdlineParams(argc, argv, DeviceCount, PM);
}
