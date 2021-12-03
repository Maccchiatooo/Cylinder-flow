#ifndef _LBM_H_
#define _LBM_H_

#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>

#define q 9
#define dim 2

class LBM
{
    typedef Kokkos::RangePolicy<> range_policy;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy3;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy2;

public:
    LBM(int sx, int sy, double &tau, double &rho0, double &u0) : lx(sx + 5), ly(sy + 5), tau0(tau), rho0(rho0), u0(u0){

                                                                                                                };

    void Initialize();
    void Collision();
    void Streaming();
    //  void BC();
    void Update();
    void Output(int n);

    // domain size and relaxation time
    int lx, ly;
    double rho0;
    double mu;
    double cs2;
    double tau0;

    // initial velocity
    double u0;

    Kokkos::View<double ***, Kokkos::CudaUVMSpace> f = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("f", q, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> ft = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("ft", q, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> fb = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("fb", q, lx, ly);

    Kokkos::View<double **, Kokkos::CudaUVMSpace> ua = Kokkos::View<double **, Kokkos::CudaUVMSpace>("u", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> va = Kokkos::View<double **, Kokkos::CudaUVMSpace>("v", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> rho = Kokkos::View<double **, Kokkos::CudaUVMSpace>("rho", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> p = Kokkos::View<double **, Kokkos::CudaUVMSpace>("p", lx, ly);
    Kokkos::View<int **, Kokkos::CudaUVMSpace> usr = Kokkos::View<int **, Kokkos::CudaUVMSpace>("usr", lx, ly);
    Kokkos::View<int **, Kokkos::CudaUVMSpace> ran = Kokkos::View<int **, Kokkos::CudaUVMSpace>("ran", lx, ly);

    Kokkos::View<int **, Kokkos::CudaUVMSpace> e = Kokkos::View<int **, Kokkos::CudaUVMSpace>("e", q, dim);
    Kokkos::View<double *, Kokkos::CudaUVMSpace> t = Kokkos::View<double *, Kokkos::CudaUVMSpace>("t", q);
    Kokkos::View<int *, Kokkos::CudaUVMSpace> bb = Kokkos::View<int *, Kokkos::CudaUVMSpace>("b", q);
};
#endif