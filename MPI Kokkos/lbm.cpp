#include "lbm.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>

using namespace std;

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

        gen_type rgen = rand_pool.get_state();
        for (long it = 0; it < dart_groups; ++it)
        {
            Scalar x = Kokkos::rand<gen_type, Scalar>::draw(rgen);
            Scalar y = Kokkos::rand<gen_type, Scalar>::draw(rgen);
            Scalar dSq = (x * x + y * y);

            if (dSq <= rad * rad)
            {
                ++lsum;
            }
        }

        rand_pool.free_state(rgen);
    }

    GenRandom(RandPool rand_pool_, Scalar rad_, long dart_groups_)
        : rand_pool(rand_pool_), rad(rad_), dart_groups(dart_groups_) {}

}; // end GenRandom struct

void LBM::Initialize()
{

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

            p(i, j) = 0.0;
            rho(i, j) = 1.0;
            usr(i, j) = (pow((lnx - 0.5 * (glx - 1)), 2) + pow((lny - 0.5 * (gly - 1)), 2) <= pow(0.03125 * (glx - 1) / 2.0, 2)) ? 0 : 1;
            ua(i, j) = (u0 * 4.0 * lny * (gly - 1 - lny) / pow((gly - 1), 2)) * usr(i, j) * (0.9 + ran(i, j) * 0.002);
            va(i, j) = 0.0;
            fdx(i, j) = 0.0;
            fdy(i, j) = 0.0;
        });
    // distribution function initialization
    Kokkos::fence();
    Kokkos::parallel_for(
        "initf", mdrange_policy3({0, 0, 0}, {q, lx, ly}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            f(ii, i, j) = t(ii) * p(i, j) * 3.0 +
                          t(ii) * (3.0 * (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                   4.5 * pow(e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j), 2) -
                                   1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));

            ft(ii, i, j) = 0.0;
        });
    Kokkos::fence();
};
void LBM::Collision()
{
    Kokkos::parallel_for(
        "collision", mdrange_policy3({0, 2, 2}, {q, lx - 2, ly - 2}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            double feq = t(ii) * p(i, j) * 3.0 +
                         t(ii) * (3.0 * (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                  4.5 * pow(e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j), 2) -
                                  1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));
            // ft(ii, i, j) = f(ii, i, j);
            f(ii, i, j) -= (f(ii, i, j) - feq) / (tau0 + 0.5);
        });
    Kokkos::fence();
};

void LBM::setup_subdomain()
{
    if (x_lo != 0)
        m_left = buffer_t("m_left", q, ly);

    if (x_hi != glx - 1)
        m_right = buffer_t("m_right", q, ly);

    if (y_lo != 0)
        m_down = buffer_t("m_down", q, lx);

    if (y_hi != gly - 1)
        m_up = buffer_t("m_up", q, lx);

    if (x_lo != 0 && y_hi != gly - 1)
        m_leftup = buffer_ut("m_leftup", q);

    if (x_hi != glx && y_hi != gly - 1)
        m_rightup = buffer_ut("m_rightup", q);

    if (x_lo != 0 && y_lo != 0)
        m_leftdown = buffer_ut("m_leftdown", q);

    if (x_hi != glx - 1 && y_lo != 0)
        m_rightdown = buffer_ut("m_rightdown", q);

    if (x_lo != 0)
        m_leftout = buffer_t("m_leftout", q, ly);

    if (x_hi != glx - 1)
        m_rightout = buffer_t("m_rightout", q, ly);

    if (y_lo != 0)
        m_downout = buffer_t("m_downout", q, lx);

    if (y_hi != gly - 1)
        m_upout = buffer_t("m_upout", q, lx);

    if (x_lo != 0 && y_hi != gly - 1)
        m_leftupout = buffer_ut("m_leftupout", q);

    if (x_hi != glx - 1 && y_hi != gly - 1)
        m_rightupout = buffer_ut("m_rightupout", q);

    if (x_lo != 0 && y_lo != 0)
        m_leftdownout = buffer_ut("m_leftdownout", q);

    if (x_hi != glx - 1 && y_lo != 0)
        m_rightdownout = buffer_ut("m_rightdownout", q);
}
void LBM::pack()
{
    if (x_lo != 0)
        Kokkos::deep_copy(m_leftout, Kokkos::subview(f, Kokkos::ALL, 2, Kokkos::ALL));

    if (x_hi != glx - 1)
        Kokkos::deep_copy(m_rightout, Kokkos::subview(f, Kokkos::ALL, lx - 3, Kokkos::ALL));

    if (y_lo != 0)
        Kokkos::deep_copy(m_downout, Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, 2));

    if (y_hi != gly - 1)
        Kokkos::deep_copy(m_upout, Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, ly - 3));

    if (x_lo != 0 && y_hi != gly - 1)
        Kokkos::deep_copy(m_leftupout, Kokkos::subview(f, Kokkos::ALL, 2, ly - 3));

    if (x_hi != glx - 1 && y_hi != gly - 1)
        Kokkos::deep_copy(m_rightupout, Kokkos::subview(f, Kokkos::ALL, lx - 3, ly - 3));

    if (x_lo != 0 && y_lo != 0)
        Kokkos::deep_copy(m_leftdownout, Kokkos::subview(f, Kokkos::ALL, 2, 2));

    if (x_hi != glx - 1 && y_lo != 0)
        Kokkos::deep_copy(m_rightdownout, Kokkos::subview(f, Kokkos::ALL, lx - 3, 2));

    Kokkos::fence();
}

void LBM::exchange()
{
    int mar = 1;

    if (x_lo != 0)
        MPI_Send(m_leftout.data(), m_leftout.size(), MPI_DOUBLE, comm.left, mar, comm.comm);

    if (x_hi != glx - 1)
        MPI_Recv(m_right.data(), m_right.size(), MPI_DOUBLE, comm.right, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 2;
    if (x_hi != glx - 1)
        MPI_Send(m_rightout.data(), m_rightout.size(), MPI_DOUBLE, comm.right, mar, comm.comm);

    if (x_lo != 0)
        MPI_Recv(m_left.data(), m_left.size(), MPI_DOUBLE, comm.left, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 3;

    if (y_lo != 0)
        MPI_Send(m_downout.data(), m_downout.size(), MPI_DOUBLE, comm.down, mar, comm.comm);

    if (y_hi != gly - 1)
        MPI_Recv(m_up.data(), m_up.size(), MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 4;
    if (y_hi != gly - 1)
        MPI_Send(m_upout.data(), m_upout.size(), MPI_DOUBLE, comm.up, mar, comm.comm);

    if (y_lo != 0)
        MPI_Recv(m_down.data(), m_down.size(), MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 5;

    if (x_lo != 0 && y_hi != gly - 1)
        MPI_Send(m_leftupout.data(), m_leftupout.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm);

    if (x_hi != glx - 1 && y_lo != 0)
        MPI_Recv(m_rightdown.data(), m_rightdown.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 6;
    if (x_hi != glx - 1 && y_hi != gly - 1)
        MPI_Send(m_rightupout.data(), m_rightupout.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0)
        MPI_Recv(m_leftdown.data(), m_leftdown.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 7;
    if (x_lo != 0 && y_lo != 0)
        MPI_Send(m_leftdownout.data(), m_leftdownout.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm);

    if (x_hi != glx - 1 && y_hi != gly - 1)
        MPI_Recv(m_rightup.data(), m_rightup.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 8;
    if (x_hi != glx - 1 && y_lo != 0)
        MPI_Send(m_rightdownout.data(), m_rightdownout.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly - 1)
        MPI_Recv(m_leftup.data(), m_leftup.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);
}

void LBM::unpack()
{
    if (x_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, 1, Kokkos::ALL), m_left);

    if (x_hi != glx - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - 2, Kokkos::ALL), m_right);

    if (y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, 1), m_down);

    if (y_hi != gly - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, Kokkos::ALL, ly - 2), m_up);

    if (x_lo != 0 && y_hi != gly - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, 1, ly - 2), m_leftup);

    if (x_hi != glx - 1 && y_hi != gly - 1)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - 2, ly - 2), m_rightup);

    if (x_lo != 0 && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, 1, 1), m_leftdown);

    if (x_hi != glx - 1 && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - 2, 1), m_rightdown);

    Kokkos::fence();
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

    if (x_hi == glx - 1)
        Kokkos::parallel_for(
            "stream3", range_policy(1, ly - 1), KOKKOS_CLASS_LAMBDA(const int j) {
                f(3, lx - 2, j) = f(3, lx - 3, j);
                f(7, lx - 2, j) = f(7, lx - 3, j + 1);
                f(6, lx - 2, j) = f(6, lx - 3, j - 1);
            });

    if (y_lo == 0)
        Kokkos::parallel_for(
            "stream4", range_policy(1, lx - 1), KOKKOS_CLASS_LAMBDA(const int i) {
                f(2, i, 1) = f(4, i, 3);
                f(5, i, 1) = f(7, i + 2, 3);
                f(6, i, 1) = f(8, i - 2, 3);
            });

    if (y_hi == gly - 1)
        Kokkos::parallel_for(
            "stream4", range_policy(1, lx - 1), KOKKOS_CLASS_LAMBDA(const int i) {
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
    Kokkos::fence();

    Kokkos::parallel_for(
        "initv", mdrange_policy2({0, 0}, {lx, ly}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            fdx(i, j) = 0.0;
            fdy(i, j) = 0.0;
        });

    Kokkos::fence();

    for (int i = 2; i < lx - 2; i++)
    {
        for (int j = 2; j < ly - 2; j++)
        {
            for (int ii = 0; ii < q; ii++)
            {
                if (usr(i, j) == 1 && usr(i + e(ii, 0), j + e(ii, 1)) == 0)
                {
                    fdx(i, j) += (f(ii, i, j) + f(bb(ii), i - e(bb(ii), 0), j - e(bb(ii), 1))) * e(ii, 0);
                    fdy(i, j) += (f(ii, i, j) + f(bb(ii), i - e(bb(ii), 0), j - e(bb(ii), 1))) * e(ii, 1);
                }
            }
        }
    }

    Kokkos::fence();
    double resultd;
    parallel_reduce(
        " Label", mdrange_policy2({2, 2}, {lx - 2, ly - 2}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = fdx(i,j);
          valueToUpdate += my_value; }, resultd);
    Kokkos::fence();
    double resultl;
    parallel_reduce(
        " Label", mdrange_policy2({2, 2}, {lx - 2, ly - 2}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = fdy(i,j);
          valueToUpdate += my_value; }, resultl);

    Kokkos::fence();
    double sum1, sum2;
    MPI_Reduce(&resultd, &sum1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&resultl, &sum2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    ofstream outfile;
    if (comm.me == 0)
    {

        outfile.open("test.dat", ios::out | ios::app);

        outfile << std::fixed << std::setprecision(17) << 2.0 * sum1 / u0 / u0 / 0.03125 / (glx - 1) << " " << 2.0 * sum2 / u0 / u0 / 0.03125 / (glx - 1) << endl;

        outfile.close();
    }
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
            ua(i, j) = 0.0;
            va(i, j) = 0.0;
            p(i, j) = 0.0;
        });

    Kokkos::fence();

    for (int i = 2; i < lx - 2; i++)
    {
        for (int j = 2; j < ly - 2; j++)
        {
            for (int ii = 0; ii < q; ii++)
            {
                p(i, j) = p(i, j) + f(ii, i, j) / 3.0;
                ua(i, j) = ua(i, j) + f(ii, i, j) * e(ii, 0);
                va(i, j) = va(i, j) + f(ii, i, j) * e(ii, 1);
            }
            p(i, j) = p(i, j) * usr(i, j);
            ua(i, j) = ua(i, j) * usr(i, j);
            va(i, j) = va(i, j) * usr(i, j);
            if (x_lo == 0)
            {
                ua(2, j) = (u0 * 4.0 * (y_lo + j - 2) * (gly - 1 - (y_lo + j - 2)) / pow((gly - 1), 2));
                va(2, j) = 0.0;
            }
            if (x_hi == glx - 1)
            {
                p(lx - 3, j) = 0.0;
            }
        }
    }

    Kokkos::fence();
};

void LBM::MPIoutput(int n)
{
    // MPI_IO
    MPI_File fh;
    MPI_Status status;
    MPI_Offset offset = 0;

    MPI_Datatype FILETYPE, DATATYPE;
    // buffer
    int tp;
    float ttp;
    double fp;
    // min max
    double umin, umax, vmin, vmax, pmin, pmax;
    double uumin, uumax, vvmin, vvmax, ppmin, ppmax;
    // transfer
    double *uu, *vv, *pp, *xx, *yy;
    uu = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));
    vv = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));
    pp = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));
    xx = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));
    yy = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));

    for (int j = 0; j < (ly - 2 * ghost); j++)
    {
        for (int i = 0; i < (lx - 2 * ghost); i++)
        {

            uu[i + j * (lx - 2 * ghost)] = ua(i + ghost, j + ghost);
            vv[i + j * (lx - 2 * ghost)] = va(i + ghost, j + ghost);
            pp[i + j * (lx - 2 * ghost)] = p(i + ghost, j + ghost);
            xx[i + j * (lx - 2 * ghost)] = (double)4.0 * (x_lo + i) / (glx - 1);
            yy[i + j * (lx - 2 * ghost)] = (double)(y_lo + j) / (gly - 1);
        }
    }

    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = ua(i,j);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(umax));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = va(i,j);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(vmax));
    Kokkos::fence();

    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = p(i,j);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(pmax));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = ua(i,j);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(umin));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = va(i,j);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(vmin));
    Kokkos::fence();

    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = p(i,j);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(pmin));
    Kokkos::fence();
    std::string str1 = "output" + std::to_string(n) + ".plt";
    const char *na = str1.c_str();
    std::string str2 = "#!TDV112";
    const char *version = str2.c_str();
    MPI_File_open(MPI_COMM_WORLD, na, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_Reduce(&umin, &uumin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&umax, &uumax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&vmin, &vvmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&vmax, &vvmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&pmin, &ppmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pmax, &ppmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (comm.me == 0)
    {

        MPI_File_seek(fh, offset, MPI_SEEK_SET);
        // header !version number
        MPI_File_write(fh, version, 8, MPI_CHAR, &status);
        // INTEGER 1
        tp = 1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 3*4+8=20
        // variable name
        tp = 5;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 120;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 121;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 117;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 118;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 112;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 20+11*4=64
        // Zone Marker
        ttp = 299.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // Zone Name
        tp = 90;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 79;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 78;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 69;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 32;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 48;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 48;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 49;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 64 + 10 * 4 = 104

        // paraents
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // strendid
        tp = -2;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SOLUTION TIME
        double nn = (double)n;
        fp = nn;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        // zone color
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE type
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // specify var location
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // are raw local
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // number of miscellaneous
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ordered zone
        tp = 0;

        tp = glx;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = gly;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // AUXILIARY
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 104 + 13 * 4 = 156
        // EOHMARKER
        ttp = 357.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // DATA SECTION
        ttp = 299.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // VARIABLE DATA FORMAT
        tp = 2;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // PASSIVE VARIABLE
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SHARING VARIABLE
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE NUMBER
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 156 + 10 * 4 = 196
        // MIN AND MAX VALUE FLOAT 64
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 4.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 1.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = uumin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = uumax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vvmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vvmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = ppmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = ppmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);

        // 196 + 10 * 8 = 276
    }

    offset = 276;

    int glolen[2] = {glx, gly};
    int localstart[2] = {x_lo, y_lo};
    int l_l[2] = {lx - 2 * ghost, ly - 2 * ghost};
    MPI_Type_create_subarray(dim, glolen, l_l, localstart, MPI_ORDER_FORTRAN, MPI_DOUBLE, &DATATYPE);

    MPI_Type_commit(&DATATYPE);

    MPI_Type_contiguous(5, DATATYPE, &FILETYPE);

    MPI_Type_commit(&FILETYPE);

    MPI_File_set_view(fh, offset, MPI_DOUBLE, FILETYPE, "native", MPI_INFO_NULL);

    MPI_File_write_all(fh, xx, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, yy, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, uu, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, vv, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, pp, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);

    free(uu);
    free(vv);
    free(pp);
    free(xx);
    free(yy);

    MPI_Barrier(MPI_COMM_WORLD);
};
void LBM::Output(int n)
{
    std::ofstream outfile;
    std::string str = "output" + std::to_string(n) + std::to_string(comm.me);
    outfile << std::setiosflags(std::ios::fixed);
    outfile.open(str + ".dat", std::ios::out);

    outfile << "variables=x,y,u,v,p" << std::endl;
    outfile << "zone I=" << lx - 4 << ",J=" << ly - 4 << std::endl;

    for (int j = 2; j < ly - 2; j++)
    {
        for (int i = 2; i < lx - 2; i++)
        {

            outfile << std::setprecision(8) << setiosflags(std::ios::left) << (i - 2.0 + x_lo) / (glx - 1.0) << " " << (j - 2.0 + y_lo) / (gly - 1.0) << " " << ua(i, j) << " " << va(i, j) << " " << p(i, j) << std::endl;
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
