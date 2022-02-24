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
#include <mpi.h>

#define q 27
#define dim 3

struct CommHelper
{

    MPI_Comm comm;
    int rx, ry, rz;
    int me;
    int px, py, pz;
    int up, down, left, right, front, back, frontup, frontdown, frontleft, frontright, frontleftup, frontleftdown, frontrightup, frontrightdown, backup, backdown, backleft, backright, backleftup, backrightup, backleftdown, backrightdown, leftup, leftdown, rightup, rightdown;

    CommHelper(MPI_Comm comm_)
    {
        comm = comm_;
        int nranks;
        MPI_Comm_size(comm, &nranks);
        MPI_Comm_rank(comm, &me);

        rx = std::pow(1.0 * nranks, 1.0 / 3.0);
        while (nranks % rx != 0)
            rx++;

        rz = std::sqrt(1.0 * (nranks / rx));
        while ((nranks / rx) % rz != 0)
            rz++;

        ry = nranks / rx / rz;

        // printf("rx=%i,ry=%i,rz=%i\n", rx, ry, rz);
        px = me % rx;
        pz = (me / rx) % rz;
        py = (me / rx / rz);

        left = px == 0 ? -1 : me - 1;
        leftup = (px == 0 || pz == rz - 1) ? -1 : me - 1 + rx;
        rightup = (px == rx - 1 || pz == rz - 1) ? -1 : me + 1 + rx;
        leftdown = (px == 0 || pz == 0) ? -1 : me - 1 - rx;
        rightdown = (px == rx - 1 || pz == 0) ? -1 : me + 1 - rx;
        right = px == rx - 1 ? -1 : me + 1;
        down = pz == 0 ? -1 : me - rx;
        up = pz == rz - 1 ? -1 : me + rx;

        front = py == 0 ? -1 : me - rx * rz;
        frontup = (py == 0 || pz == rz - 1) ? -1 : me - rx * rz + rx;
        frontdown = (py == 0 || pz == 0) ? -1 : me - rx * rz - rx;
        frontleft = (py == 0 || px == 0) ? -1 : me - rx * rz - 1;
        frontright = (py == 0 || px == rx - 1) ? -1 : me - rx * rz + 1;
        frontleftdown = (py == 0 || px == 0 || pz == 0) ? -1 : me - rx * rz - rx - 1;
        frontrightdown = (py == 0 || px == rx - 1 || pz == 0) ? -1 : me - rx * rz - rx + 1;
        frontrightup = (py == 0 || px == rx - 1 || pz == rz - 1) ? -1 : me - rx * rz + rx + 1;
        frontleftup = (py == 0 || px == 0 || pz == rz - 1) ? -1 : me - rx * rz + rx - 1;

        back = py == ry - 1 ? -1 : me + rx * rz;
        backup = (py == ry - 1 || pz == rz - 1) ? -1 : me + rx * rz + rx;
        backdown = (py == ry - 1 || pz == 0) ? -1 : me + rx * rz - rx;
        backleft = (py == ry - 1 || px == 0) ? -1 : me + rx * rz - 1;
        backright = (py == ry - 1 || px == rx - 1) ? -1 : me + rx * rz + 1;
        backleftdown = (py == ry - 1 || px == 0 || pz == 0) ? -1 : me + rx * rz - rx - 1;
        backrightdown = (py == ry - 1 || px == rx - 1 || pz == 0) ? -1 : me + rx * rz - rx + 1;
        backrightup = (py == ry - 1 || px == rx - 1 || pz == rz - 1) ? -1 : me + rx * rz + rx + 1;
        backleftup = (py == ry - 1 || px == 0 || pz == rz - 1) ? -1 : me + rx * rz + rx - 1;

        // printf("Me:%i MyNeibors: %i %i %i %i %i %i %i %i\n", me, left, right, up, down, leftup, leftdown, rightup, rightdown);
        // printf("Me:%i MyfrontNeibors: %i %i %i %i %i %i %i %i\n", front, frontleft, frontright, frontup, frontdown, frontleftup, frontleftdown, frontrightup, frontrightdown);
        // printf("Me:%i MybackNeibors: %i %i %i %i %i %i %i %i\n", back, backleft, backright, backup, backdown, backleftup, backleftdown, backrightup, backrightdown);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    template <class ViewType>
    void isend_irecv(int partner, ViewType send_buffer, ViewType recv_buffer, MPI_Request *request_send, MPI_Request *request_recv)
    {
        MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_recv);
        MPI_Isend(send_buffer.data(), send_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_send);
    }
};
struct LBM
{
    typedef Kokkos::RangePolicy<> range_policy;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy3;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<4>> mdrange_policy4;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy2;
    using buffer_t = Kokkos::View<double ***, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace>;
    using buffer_ut = Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace>;
    using buffer_st = Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace>;
    CommHelper comm;
    MPI_Request mpi_requests_recv[26];
    MPI_Request mpi_requests_send[26];
    int mpi_active_requests;

    int glx;
    int gly;
    int glz;
    int ghost = 3;
    // exact nodes
    int ex = glx / comm.rx;
    int ey = gly / comm.ry;
    int ez = glz / comm.rz;
    // include ghost nodes
    int lx = ex + 2 * ghost;
    int ly = ey + 2 * ghost;
    int lz = ez + 2 * ghost;

    int x_lo, x_hi, y_lo, y_hi, z_lo, z_hi;
    double rho0;
    double mu;
    double cs2;
    double tau0;
    double u0;

    buffer_t m_left, m_right, m_down, m_up, m_front, m_back;
    buffer_t m_leftout, m_rightout, m_downout, m_upout, m_frontout, m_backout;
    buffer_ut m_leftup, m_rightup, m_leftdown, m_rightdown, m_frontup, m_backup, m_frontdown, m_backdown, m_frontleft, m_backleft, m_frontright, m_backright;
    buffer_ut m_leftupout, m_rightupout, m_leftdownout, m_rightdownout;
    buffer_ut m_frontupout, m_backupout, m_frontdownout, m_backdownout, m_frontleftout, m_backleftout, m_frontrightout, m_backrightout;
    buffer_st m_frontleftup, m_frontrightup, m_frontleftdown, m_frontrightdown, m_backleftup, m_backleftdown, m_backrightup, m_backrightdown;
    buffer_st m_frontleftupout, m_frontrightupout, m_frontleftdownout, m_frontrightdownout, m_backleftupout, m_backleftdownout, m_backrightupout, m_backrightdownout;

    Kokkos::View<double ****, Kokkos::CudaUVMSpace> f = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("f", q, lx, ly, lz);
    Kokkos::View<double ****, Kokkos::CudaUVMSpace> ft = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("ft", q, lx, ly, lz);
    Kokkos::View<double ****, Kokkos::CudaUVMSpace> fb = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("fb", q, lx, ly, lz);

    Kokkos::View<double ***, Kokkos::CudaUVMSpace> ua = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("u", lx, ly, lz);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> va = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("v", lx, ly, lz);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> wa = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("v", lx, ly, lz);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> rho = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("rho", lx, ly, lz);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> p = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("p", lx, ly, lz);

    Kokkos::View<int **, Kokkos::CudaUVMSpace> e = Kokkos::View<int **, Kokkos::CudaUVMSpace>("e", q, dim);
    Kokkos::View<double *, Kokkos::CudaUVMSpace> t = Kokkos::View<double *, Kokkos::CudaUVMSpace>("t", q);
    Kokkos::View<int ***, Kokkos::CudaUVMSpace> usr = Kokkos::View<int ***, Kokkos::CudaUVMSpace>("usr", lx, ly, lz);
    Kokkos::View<int ***, Kokkos::CudaUVMSpace> ran = Kokkos::View<int ***, Kokkos::CudaUVMSpace>("ran", lx, ly, lz);
    Kokkos::View<int *, Kokkos::CudaUVMSpace> bb = Kokkos::View<int *, Kokkos::CudaUVMSpace>("b", q);

    LBM(MPI_Comm comm_, int sx, int sy, int sz, double &tau, double &rho0, double &u0) : comm(comm_), glx(sx), gly(sy), glz(sz), tau0(tau), rho0(rho0), u0(u0){
                                                                                                                                                            // if (comm.me == 0)
                                                                                                                                                            //  printf("rho=%f,tau0=%f,u0=%f\n", rho0, tau0, u0);
                                                                                                                                                        };

    void Initialize();
    void Collision();
    void setup_subdomain();
    void pack();
    void exchange();
    void unpack();
    void Streaming();
    void Update();
    void MPIoutput(int n);
    void Output(int n);
};
#endif