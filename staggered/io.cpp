#include <stdio.h>
#include <fstream>
#include "CLI11.hpp"
#include "data_global.h"

void data_ext::DumpData(const char* fname, const real3* velocity, const real* pressure, const real* qressure) const{
    const int N[3] = {NX, NY, NZ};
    const int NN = N[0]*N[1]*N[2];
    FILE* f = fopen(fname, "wt");
    fprintf(f, "# vtk DataFile Version 2.0\n");
    fprintf(f, "Volume example\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET RECTILINEAR_GRID\n");
    fprintf(f, "DIMENSIONS %i %i %i\n", N[0], N[1], N[2]);
    for(int idir=0; idir<3; idir++){
        fprintf(f, "%c_COORDINATES %i float\n", 'X'+idir, N[idir]);
        for(int i=0; i<N[idir]; i++) fprintf(f, "%e ", (idir==0) ? xmin+hx*i : (idir==1) ? ymin+hy*i : ZZ[i]);
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

int data_ext::ReadData(const char* fname, real3* velocity, real* pressure) const{
    const int N[3] = {NX, NY, NZ};
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

template<typename fpv>
void data_ext::DumpProfiles(const real9<fpv>* av, const char* fname){
    const real u_fric = 1;
    FILE* F = fopen(fname, "wt");

    for(int iz=0; iz<NZ; iz++){
        double H = 0.5*(ZZ[NZ] - ZZ[0]); // channel half-width
        double z = 0.5*(ZZ[iz]+ZZ[iz+1]) - ZZ[0]; // coordinate of the point the averaged data belongs to
        double dist_to_wall = (z<H) ? z : 2.*H-z;
        double y_plus = dist_to_wall*u_fric / visc;

        double u[3], uu[6];
        for(int i=0; i<3; i++) u[i] = av[iz].u[i] / u_fric;
        for(int i=0; i<6; i++) uu[i] = av[iz].uu[i] / (u_fric*u_fric);
        uu[0] = std::max(0., uu[0]-u[0]*u[0]);
        uu[1] = std::max(0., uu[1]-u[1]*u[1]);
        uu[2] = std::max(0., uu[2]-u[2]*u[2]);
        uu[3] -= u[0]*u[1];
        uu[4] -= u[0]*u[2];
        uu[5] -= u[1]*u[2];

        double nu_t = av[iz].nu_t;
        double pp = std::max(0., double(av[iz].pp) - av[iz].p*av[iz].p);
        fprintf(F, "%e %e %e %e %e %e %e %e\n", y_plus, u[0], sqrt(uu[0]), sqrt(uu[1]), sqrt(uu[2]), uu[4], nu_t/visc, sqrt(pp));
    }
    fclose(F);
}

template void data_ext::DumpProfiles(const real9<double>* av, const char* fname);
template void data_ext::DumpProfiles(const real9<float>* av, const char* fname);


int ReadCmdlineParams(int argc, char** argv, int DeviceCount, t_cmdline_params& PM){
    CLI::App app{"Turbulent flow between parallel plates"};
    argv = app.ensure_utf8(argv);

    std::string filename = "default";
    app.add_option("-d,--device", PM.DeviceID, std::string("device to use, 0..") + std::to_string(DeviceCount-1));
    app.add_option("--nz", PM.NZ, "number of nodes in the normal direction");
    app.add_option("-t,--TimeMax", PM.TimeMax, "maximal integration time");
    app.add_option("-a,--TimeStartAveraging", PM.TimeStartAveraging, "time where averaging should be started");
    app.add_option("--TimeStepsMax", PM.TimeStepsMax, "maximal number of iterations");
    app.add_option("--tau", PM.tau, "initial timestep size");
    app.add_option("--CFLmin", PM.CFLmin, "if CFL falls below this value, then the timestep will be increased");
    app.add_option("--CFLmax", PM.CFLmax, "if CFL lifts above this value, then the timestep will be decreased");
    app.add_option("-s,--SProgress", PM.SProgress, "number of timesteps between writing to screen and log");
    app.add_option("-f,--FProgress", PM.FProgress, "number of timesteps between writing a field");
    app.add_option("-p,--PProgress", PM.PProgress, "number of timesteps between writing a profile");

    try {
        app.parse(argc, argv);
    } catch(const CLI::ParseError &e) {
        return 1234567+app.exit(e); // print error (or help) message and return a nonzero code
    }
    return 0; // success
}
