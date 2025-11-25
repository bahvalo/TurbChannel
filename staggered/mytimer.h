#ifndef MYTIMER_H
#define MYTIMER_H
#include <chrono>

struct MyTimer{
    using time_type = std::chrono::time_point<std::chrono::steady_clock>;
    time_type started;
    double timer = 0.;
    void beg(){
        started = std::chrono::steady_clock::now();
    }
    void end(){
        time_type finished = std::chrono::steady_clock::now();
        const std::chrono::duration<double> diff = finished - started;
        timer += diff.count();
    };
};

extern MyTimer t_ExplicitTerm, t_ApplyGradient, t_PressureSolver[4]; // Stuff defined in func.cpp

#endif
