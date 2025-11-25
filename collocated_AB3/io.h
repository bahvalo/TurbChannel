#ifndef IO_H
#define IO_H

// IO functions
void DumpData(const main_params& MD, const double* Z, const char* fname, const real3* velocity, const real* pressure, const real* qressure);
int ReadData(const main_params& MD, const char* fname, real3* velocity, real* pressure);
void DumpProfiles(const main_params& MD, const double* Z, const real9* av, double visc, double u_fric, const char* fname, int IsZplus=0);


struct t_cmdline_params{
    int DeviceID = 0; // if there are several GPUs, select one
    double TimeMax = 100.; // maximal integration time
    double TimeStartAveraging = 0.; // time where averaging should be started
    int TimeStepsMax = 0x7FFFFFFF; // maximal number of iterations
    double tau = 1e-3; // initial timestep size
    double CFLmin = 0.05; // if CFL falls below this value, then the timestep will be increased
    double CFLmax = 0.5; // if CFL lifts above this value, then the timestep will be decreased
    int SProgress = 10; // number of timesteps between writing to screen and log
    int FProgress = 10000; // number of timesteps between writing a field
    int PProgress = 100000; // number of timesteps between writing a profile
};
int ReadCmdlineParams(int argc, char** argv, int DeviceCount, t_cmdline_params& PM);

#endif
