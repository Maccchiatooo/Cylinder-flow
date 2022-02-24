#ifndef _SYSTEM_H_
#define _SYSTEM_H_
#include "lbm.hpp"
#include <cmath>
#include <iostream>

class System
{

public:
    System(int &nx, int &ny, int &nz) : sx(nx), sy(ny), sz(nz){};
    void Initialize();
    void Monitor();

    //  domain size
    int sx, sy, sz;
    // viscosity
    double miu;
    // Renolds number
    double Re;
    // input velocity
    double u0;
    // circle radius
    int R;
    // density
    double rho0;
    // speed of sound
    double cs2;
    double cs;
    // Mach number
    double Ma;
    // total time
    int Time;
    // time interval
    int inter;
    // relaxation time connect to macro to micro
    double tau;
};
#endif