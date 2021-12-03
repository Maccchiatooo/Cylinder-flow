#include "lbm.hpp"
#include "System.hpp"
#include <Kokkos_Core.hpp>

int main(int argc, char *argv[])
{
    int nx = 512;
    int ny = 128;

    Kokkos::initialize(argc, argv);
    {

        System s1(nx, ny);

        s1.Initialize();
        s1.Monitor();

        LBM l1(s1.sx, s1.sy, s1.tau, s1.rho0, s1.u0);

        l1.Initialize();
        l1.Output(0);

        for (int it = 1; it <= s1.Time; it++)
        {

            l1.Collision();
            l1.Streaming();
            l1.Update();

            if (it % s1.inter == 0)
            {
                l1.Output(it / s1.inter);
            }
        }
    }
    Kokkos::finalize();

    return 0;
}