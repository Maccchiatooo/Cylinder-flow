#include "lbm.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>

template <class RandPool>
struct GenRandom
{

    // The GeneratorPool
    RandPool rand_pool;

    typedef double Scalar;
    typedef typename RandPool::generator_type gen_type;

    Scalar rad;       // target radius and box size
    long dart_groups; // Reuse the generator for drawing random #s this many times

    KOKKOS_INLINE_FUNCTION
    void operator()(long i, long &lsum) const
    {
        // Get a random number state from the pool for the active thread
        gen_type rgen = rand_pool.get_state();

        // Draw samples numbers from the pool as urand64 between 0 and rand_pool.MAX_URAND64
        // Note there are function calls to get other type of scalars, and also to specify
        // Ranges or get a normal distributed float.
        for (long it = 0; it < dart_groups; ++it)
        {
            Scalar x = Kokkos::rand<gen_type, Scalar>::draw(rgen);
            Scalar y = Kokkos::rand<gen_type, Scalar>::draw(rgen);

            // Example - if you wish to draw from a normal distribution
            // mean = .5, stddev = 0.125
            // Scalar x = rgen.normal(.5,.125);
            // Scalar y = rgen.normal(.5,.125);

            Scalar dSq = (x * x + y * y);

            if (dSq <= rad * rad)
            {
                ++lsum;
            } // comparing to rad^2 - am I in the circle inscribed in square?
        }

        // Give the state back, which will allow another thread to aquire it
        rand_pool.free_state(rgen);
    }

    // Constructor, Initialize all members
    GenRandom(RandPool rand_pool_, Scalar rad_, long dart_groups_)
        : rand_pool(rand_pool_), rad(rad_), dart_groups(dart_groups_) {}

}; // end GenRandom struct

void LBM::Initialize()
{

    x_lo = (lx - 4) * comm.px;
    x_hi = (lx - 4) * (comm.px + 1);
    y_lo = (ly - 4) * comm.py;
    y_hi = (ly - 4) * (comm.py + 1);

    printf("Me=%i,x_lo=%i,y_lo=%i,x_hi=%i,y_hi=%i\n", comm.me, x_lo, y_lo, x_hi, y_hi);
    // weight and discrete velocity
    t(0) = 4.0 / 9.0;
    t(1) = 1.0 / 9.0;
    t(2) = 1.0 / 9.0;
    t(3) = 1.0 / 9.0;
    t(4) = 1.0 / 9.0;
    t(5) = 1.0 / 36.0;
    t(6) = 1.0 / 36.0;
    t(7) = 1.0 / 36.0;
    t(8) = 1.0 / 36.0;

    bb(0) = 0;
    bb(1) = 3;
    bb(3) = 1;
    bb(2) = 4;
    bb(4) = 2;
    bb(5) = 7;
    bb(7) = 5;
    bb(6) = 8;
    bb(8) = 6;

    e(0, 0) = 0;
    e(1, 0) = 1;
    e(2, 0) = 0;
    e(3, 0) = -1;
    e(4, 0) = 0;
    e(5, 0) = 1;
    e(6, 0) = -1;
    e(7, 0) = -1;
    e(8, 0) = 1;

    e(0, 1) = 0;
    e(1, 1) = 0;
    e(2, 1) = 1;
    e(3, 1) = 0;
    e(4, 1) = -1;
    e(5, 1) = 1;
    e(6, 1) = 1;
    e(7, 1) = -1;
    e(8, 1) = -1;

    typedef typename Kokkos::Random_XorShift64_Pool<> RandPoolType;
    RandPoolType rand_pool(5374857);
    Kokkos::parallel_for(
        "ran", mdrange_policy2({0, 0}, {lx, ly}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            auto state = rand_pool.get_state();
            ran(i, j) = state.urand(100);
            rand_pool.free_state(state);
        });

    Kokkos::parallel_for(
        "initv", mdrange_policy2({0, 0}, {lx, ly}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            int lnx = x_lo + i - 2;
            int lny = y_lo + j - 2;

            p(i, j) = 0;
            rho(i, j) = rho0;
            usr(i, j) = (pow((lnx - 0.25 * (glx - 1)), 2) + pow((lny - 0.5 * (gly - 1)), 2) <= pow(0.1 * (gly - 1), 2)) ? 0 : 1;
            ua(i, j) = (u0 * 4.0 * lny * (gly - 1 - lny) / pow((gly - 1), 2)) * usr(i, j) * (1.0 + ran(i, j) * 0.001);
            va(i, j) = 0.0;
        });
    // distribution function initialization

    Kokkos::parallel_for(
        "initf", mdrange_policy3({0, 0, 0}, {q, lx, ly}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            f(ii, i, j) = t(ii) * p(i, j) * 3.0 +
                          t(ii) * (3.0 * (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                   4.5 * pow(e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j), 2) -
                                   1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));

            ft(ii, i, j) = 0;
        });
};
void LBM::Collision()
{
    Kokkos::parallel_for(
        "collision", mdrange_policy3({0, 1, 1}, {q, lx - 1, ly - 1}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            double feq = t(ii) * p(i, j) * 3.0 +
                         t(ii) * (3.0 * (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                  4.5 * pow(e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j), 2) -
                                  1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));
            f(ii, i, j) -= (f(ii, i, j) - feq) / (tau0 + 0.5);
        });
};

void LBM::setup_subdomain()
{
    if (x_lo != 0)
        m_left = buffer_t("m_left", q, ly - 4);

    if (x_hi != glx)
        m_right = buffer_t("m_right", q, ly - 4);

    if (y_lo != 0)
        m_down = buffer_t("m_down", q, lx - 4);

    if (y_hi != gly)
        m_up = buffer_t("m_up", q, lx - 4);

    if (x_lo != 0 && y_hi != gly)
        m_leftup = buffer_ut("m_leftup", q);

    if (x_hi != glx && y_hi != gly)
        m_rightup = buffer_ut("m_rightup", q);

    if (x_lo != 0 && y_lo != 0)
        m_leftdown = buffer_ut("m_leftdown", q);

    if (x_hi != glx && y_lo != 0)
        m_rightdown = buffer_ut("m_rightdown", q);

    if (x_lo != 0)
        m_leftout = buffer_t("m_leftout", q, ly - 4);

    if (x_hi != glx)
        m_rightout = buffer_t("m_rightout", q, ly - 4);

    if (y_lo != 0)
        m_downout = buffer_t("m_downout", q, lx - 4);

    if (y_hi != gly)
        m_upout = buffer_t("m_upout", q, lx - 4);

    if (x_lo != 0 && y_hi != gly)
        m_leftupout = buffer_ut("m_leftupout", q);

    if (x_hi != glx && y_hi != gly)
        m_rightupout = buffer_ut("m_rightupout", q);

    if (x_lo != 0 && y_lo != 0)
        m_leftdownout = buffer_ut("m_leftdownout", q);

    if (x_hi != glx && y_lo != 0)
        m_rightdownout = buffer_ut("m_rightdownout", q);
}
void LBM::pack()
{
    if (x_lo != 0)
        Kokkos::deep_copy(m_leftout, Kokkos::subview(f, Kokkos::ALL, 2, std::make_pair(2, ly - 2)));

    if (x_hi != glx)
        Kokkos::deep_copy(m_rightout, Kokkos::subview(f, Kokkos::ALL, lx - 3, std::make_pair(2, ly - 2)));

    if (y_lo != 0)
        Kokkos::deep_copy(m_downout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(2, lx - 2), 2));

    if (y_hi != gly)
        Kokkos::deep_copy(m_upout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(2, lx - 2), ly - 3));

    if (x_lo != 0 && y_hi != gly)
        Kokkos::deep_copy(m_leftupout, Kokkos::subview(f, Kokkos::ALL, 2, ly - 3));

    if (x_hi != glx && y_hi != gly)
        Kokkos::deep_copy(m_rightupout, Kokkos::subview(f, Kokkos::ALL, lx - 3, ly - 3));

    if (x_lo != 0 && y_lo != 0)
        Kokkos::deep_copy(m_leftdownout, Kokkos::subview(f, Kokkos::ALL, 2, 2));

    if (x_hi != glx && y_lo != 0)
        Kokkos::deep_copy(m_rightdownout, Kokkos::subview(f, Kokkos::ALL, lx - 3, 2));
}

void LBM::exchange()
{
    int mar = 1;

    if (x_lo != 0)
        MPI_Send(m_leftout.data(), m_leftout.size(), MPI_DOUBLE, comm.left, mar, comm.comm);

    if (x_hi != glx)
        MPI_Recv(m_right.data(), m_right.size(), MPI_DOUBLE, comm.right, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 2;
    if (x_hi != glx)
        MPI_Send(m_rightout.data(), m_rightout.size(), MPI_DOUBLE, comm.right, mar, comm.comm);

    if (x_lo != 0)
        MPI_Recv(m_left.data(), m_left.size(), MPI_DOUBLE, comm.left, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 3;

    if (y_lo != 0)
        MPI_Send(m_downout.data(), m_downout.size(), MPI_DOUBLE, comm.down, mar, comm.comm);

    if (y_hi != gly)
        MPI_Recv(m_up.data(), m_up.size(), MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 4;
    if (y_hi != gly)
        MPI_Send(m_upout.data(), m_upout.size(), MPI_DOUBLE, comm.up, mar, comm.comm);

    if (y_lo != 0)
        MPI_Recv(m_down.data(), m_down.size(), MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 5;

    if (x_lo != 0 && y_hi != gly)
        MPI_Send(m_leftupout.data(), m_leftupout.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm);

    if (x_hi != glx && y_lo != 0)
        MPI_Recv(m_rightdown.data(), m_rightdown.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 6;
    if (x_hi != glx && y_hi != gly)
        MPI_Send(m_rightupout.data(), m_rightupout.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0)
        MPI_Recv(m_leftdown.data(), m_leftdown.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 7;
    if (x_lo != 0 && y_lo != 0)
        MPI_Send(m_leftdownout.data(), m_leftdownout.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm);

    if (x_hi != glx && y_hi != gly)
        MPI_Recv(m_rightup.data(), m_rightup.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 8;
    if (x_hi != glx && y_lo != 0)
        MPI_Send(m_rightdownout.data(), m_rightdownout.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly)
        MPI_Recv(m_leftup.data(), m_leftup.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm, MPI_STATUSES_IGNORE);
    Kokkos::fence();
}

void LBM::unpack()
{
    if (x_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, 1, std::make_pair(2, ly - 2)), m_left);

    if (x_hi != glx)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - 2, std::make_pair(2, ly - 2)), m_right);

    if (y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(2, lx - 2), 1), m_down);

    if (y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(2, lx - 2), ly - 2), m_up);

    if (x_lo != 0 && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, 1, ly - 2), m_leftup);

    if (x_hi != glx && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - 2, ly - 2), m_rightup);

    if (x_lo != 0 && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, 1, 1), m_leftdown);

    if (x_hi != glx && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - 2, 1), m_rightdown);
}

void LBM::Streaming()
{
    if (x_lo == 0)
        Kokkos::parallel_for(
            "stream3", range_policy(1, ly - 1), KOKKOS_CLASS_LAMBDA(const int j) {
                f(1, 1, j) = f(1, 2, j);
                f(5, 1, j) = f(5, 2, j);
                f(8, 1, j) = f(8, 2, j);
            });

    if (x_hi == glx)
        Kokkos::parallel_for(
            "stream3", range_policy(1, ly - 1), KOKKOS_CLASS_LAMBDA(const int j) {
                f(3, lx - 2, j) = f(3, lx - 3, j);
                f(7, lx - 2, j) = f(7, lx - 3, j + 1);
                f(6, lx - 2, j) = f(6, lx - 3, j - 1);
            });

    if (y_lo == 0)
        Kokkos::parallel_for(
            "stream4", range_policy(2, lx - 2), KOKKOS_CLASS_LAMBDA(const int i) {
                f(2, i, 1) = f(4, i, 3);
                f(5, i, 1) = f(7, i + 2, 3);
                f(6, i, 1) = f(8, i - 2, 3);
            });

    if (y_hi == gly)
        Kokkos::parallel_for(
            "stream4", range_policy(2, lx - 2), KOKKOS_CLASS_LAMBDA(const int i) {
                f(4, i, ly - 2) = f(2, i, ly - 4);
                f(7, i, ly - 2) = f(5, i - 2, ly - 4);
                f(8, i, ly - 2) = f(6, i + 2, ly - 4);
            });

    Kokkos::parallel_for(
        "usrbc", mdrange_policy3({0, 2, 2}, {q, lx - 2, ly - 2}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            if (usr(i, j) == 0 && usr(i + e(ii, 0), j + e(ii, 1)) == 1)
            {
                f(ii, i, j) = f(bb(ii), i + 2 * e(ii, 0), j + 2 * e(ii, 1));
            }
        });

    Kokkos::parallel_for(
        "stream1", mdrange_policy3({0, 2, 2}, {q, lx - 2, ly - 2}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            ft(ii, i, j) = f(ii, i - e(ii, 0), j - e(ii, 1));
        });

    Kokkos::parallel_for(
        "stream2", mdrange_policy3({0, 2, 2}, {q, lx - 2, ly - 2}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            f(ii, i, j) = ft(ii, i, j);
        });
    Kokkos::fence();
};

void LBM::Update()
{
    Kokkos::parallel_for(
        "initv", mdrange_policy2({0, 0}, {lx, ly}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            ua(i, j) = 0;
            va(i, j) = 0;
            p(i, j) = 0;
        });
    Kokkos::fence();
    for (int j = 2; j <= ly - 3; j++)
    {
        for (int i = 2; i <= lx - 3; i++)
        {
            int lny = y_lo + j - 2;
            for (int ii = 0; ii < q; ii++)
            {
                p(i, j) = p(i, j) + f(ii, i, j) / 3.0;

                ua(i, j) = ua(i, j) + f(ii, i, j) * e(ii, 0);
                va(i, j) = va(i, j) + f(ii, i, j) * e(ii, 1);
            }

            ua(i, j) = ua(i, j) * usr(i, j);
            va(i, j) = va(i, j) * usr(i, j);
            p(i, j) = p(i, j) * usr(i, j);
            if (x_lo == 0)
            {
                ua(2, j) = (u0 * 4.0 * lny * (gly - 1 - lny) / pow((gly - 1), 2));
                va(2, j) = 0.0;
            }
            if (x_hi == glx)
            {
                p(lx - 3, j) = 0.0;
            }
        }
    }
    Kokkos::fence();
};
void LBM::Output(int n)
{
    std::ofstream outfile;
    std::string str = "output" + std::to_string(n) + std::to_string(comm.me);
    outfile << std::setiosflags(std::ios::fixed);
    outfile.open(str + ".dat", std::ios::out);

    outfile << "variables=x,y,u,v,p" << std::endl;
    outfile << "zone I=" << lx - 4 << ",J=" << ly - 4 << std::endl;

    for (int j = 2; j <= ly - 3; j++)
    {
        for (int i = 2; i <= lx - 3; i++)
        {

            outfile << std::setprecision(8) << setiosflags(std::ios::left) << 4 * (i - 2.0 + x_lo) / (glx - 1) << " " << (j - 2.0 + y_lo) / (gly - 1) << " " << ua(i, j) << " " << va(i, j) << " " << p(i, j) << std::endl;
        }
    }

    outfile.close();
    if (comm.me == 0)
    {
        printf("\n");
        printf("The result %d is writen\n", n);
        printf("\n");
        printf("============================\n");
    }
};
