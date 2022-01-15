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

#define q 9
#define dim 2

struct CommHelper
{

    MPI_Comm comm;
    int rx, ry;
    int me;
    int px, py;
    int up, down, left, right, leftup, leftdown, rightup, rightdown;

    CommHelper(MPI_Comm comm_)
    {
        comm = comm_;
        int nranks;
        MPI_Comm_size(comm, &nranks);
        MPI_Comm_rank(comm, &me);

        ry = std::pow(1.0 * nranks, 1.0 / 2.0);
        while (nranks % ry != 0)
            ry++;

        rx = nranks / ry;

        px = me % rx;
        py = (me / rx) % ry;
        left = px == 0 ? -1 : me - 1;
        leftup = (px == 0 || py == ry - 1) ? -1 : me - 1 + rx;
        rightup = (px == rx - 1 || py == ry - 1) ? -1 : me + 1 + rx;
        leftdown = (px == 0 || py == 0) ? -1 : me - 1 - rx;
        rightdown = (px == rx - 1 || py == 0) ? -1 : me + 1 - rx;
        right = px == rx - 1 ? -1 : me + 1;
        down = py == 0 ? -1 : me - rx;
        up = py == ry - 1 ? -1 : me + rx;

        printf("Me:%i MyNeibors: %i %i %i %i %i %i %i %i\n", me, left, right, up, down, leftup, leftdown, rightup, rightdown);
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
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy2;
    using buffer_t = Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace>;
    using buffer_ut = Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace>;

    CommHelper comm;
    MPI_Request mpi_requests_recv[8];
    MPI_Request mpi_requests_send[8];
    int mpi_active_requests;

    int glx;
    int gly;
    int lx = glx / comm.rx + 4;
    int ly = gly / comm.ry + 4;

    int x_lo, x_hi, y_lo, y_hi;
    double rho0;
    double mu;
    double cs2;
    double tau0;
    double u0;

    buffer_t m_left, m_right, m_down, m_up;
    buffer_t m_leftout, m_rightout, m_downout, m_upout;
    buffer_ut m_leftup, m_rightup, m_leftdown, m_rightdown;
    buffer_ut m_leftupout, m_rightupout, m_leftdownout, m_rightdownout;
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> f = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("f", q, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> ft = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("ft", q, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> fb = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("fb", q, lx, ly);

    Kokkos::View<double **, Kokkos::CudaUVMSpace> ua = Kokkos::View<double **, Kokkos::CudaUVMSpace>("u", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> va = Kokkos::View<double **, Kokkos::CudaUVMSpace>("v", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> rho = Kokkos::View<double **, Kokkos::CudaUVMSpace>("rho", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> p = Kokkos::View<double **, Kokkos::CudaUVMSpace>("p", lx, ly);

    Kokkos::View<int **, Kokkos::CudaUVMSpace> e = Kokkos::View<int **, Kokkos::CudaUVMSpace>("e", q, dim);
    Kokkos::View<double *, Kokkos::CudaUVMSpace> t = Kokkos::View<double *, Kokkos::CudaUVMSpace>("t", q);
    Kokkos::View<int **, Kokkos::CudaUVMSpace> usr = Kokkos::View<int **, Kokkos::CudaUVMSpace>("usr", lx, ly);
    Kokkos::View<int **, Kokkos::CudaUVMSpace> ran = Kokkos::View<int **, Kokkos::CudaUVMSpace>("ran", lx, ly);
    Kokkos::View<int *, Kokkos::CudaUVMSpace> bb = Kokkos::View<int *, Kokkos::CudaUVMSpace>("b", q);

    LBM(MPI_Comm comm_, int sx, int sy, double &tau, double &rho0, double &u0) : comm(comm_), glx(sx), gly(sy), tau0(tau), rho0(rho0), u0(u0){

                                                                                                                                       };

    void Initialize();
    void Collision();
    void setup_subdomain();
    void pack();
    void exchange();
    void unpack();
    void Streaming();
    void Update();
    void Output(int n);
};
#endif