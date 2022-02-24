#include "lbm.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>

template <class RandPool>
struct GenRandom
{

    // The GeneratorPool
    RandPool rand_pool;

    typedef double Scalar;
    typedef typename RandPool::generator_type gen_type;

    Scalar rad;
    long dart_groups;

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
};

void LBM::Initialize()
{
    // index bound values for different cpu cores
    x_lo = ex * comm.px;
    x_hi = ex * (comm.px + 1);
    y_lo = ey * comm.py;
    y_hi = ey * (comm.py + 1);
    z_lo = ez * comm.pz;
    z_hi = ez * (comm.pz + 1);

    // printf("Me=%i,x_lo=%i,y_lo=%i,z_lo=%i,x_hi=%i,y_hi=%i,z_hi=%i\n", comm.me, x_lo, y_lo, z_lo, x_hi, y_hi, z_hi);
    //  weight function
    t(0) = 8.0 / 27.0;
    t(1) = 2.0 / 27.0;
    t(2) = 2.0 / 27.0;
    t(3) = 2.0 / 27.0;
    t(4) = 2.0 / 27.0;
    t(5) = 2.0 / 27.0;
    t(6) = 2.0 / 27.0;
    t(7) = 1.0 / 54.0;
    t(8) = 1.0 / 54.0;
    t(9) = 1.0 / 54.0;
    t(10) = 1.0 / 54.0;
    t(11) = 1.0 / 54.0;
    t(12) = 1.0 / 54.0;
    t(13) = 1.0 / 54.0;
    t(14) = 1.0 / 54.0;
    t(15) = 1.0 / 54.0;
    t(16) = 1.0 / 54.0;
    t(17) = 1.0 / 54.0;
    t(18) = 1.0 / 54.0;
    t(19) = 1.0 / 216.0;
    t(20) = 1.0 / 216.0;
    t(21) = 1.0 / 216.0;
    t(22) = 1.0 / 216.0;
    t(23) = 1.0 / 216.0;
    t(24) = 1.0 / 216.0;
    t(25) = 1.0 / 216.0;
    t(26) = 1.0 / 216.0;
    // bounce back directions
    bb(0) = 0;
    bb(1) = 2;
    bb(2) = 1;
    bb(3) = 4;
    bb(4) = 3;
    bb(5) = 6;
    bb(6) = 5;
    bb(7) = 8;
    bb(8) = 7;
    bb(9) = 10;
    bb(10) = 9;
    bb(11) = 12;
    bb(12) = 11;
    bb(13) = 14;
    bb(14) = 13;
    bb(15) = 16;
    bb(16) = 15;
    bb(17) = 18;
    bb(18) = 17;
    bb(19) = 20;
    bb(20) = 19;
    bb(21) = 22;
    bb(22) = 21;
    bb(23) = 24;
    bb(24) = 23;
    bb(25) = 26;
    bb(26) = 25;

    // discrete velocity
    e(0, 0) = 0;
    e(0, 1) = 0;
    e(0, 2) = 0;

    e(1, 0) = 1;
    e(1, 1) = 0;
    e(1, 2) = 0;

    e(2, 0) = -1;
    e(2, 1) = 0;
    e(2, 2) = 0;

    e(3, 0) = 0;
    e(3, 1) = 1;
    e(3, 2) = 0;

    e(4, 0) = 0;
    e(4, 1) = -1;
    e(4, 2) = 0;

    e(5, 0) = 0;
    e(5, 1) = 0;
    e(5, 2) = 1;

    e(6, 0) = 0;
    e(6, 1) = 0;
    e(6, 2) = -1;

    e(7, 0) = 1;
    e(7, 1) = 1;
    e(7, 2) = 0;

    e(8, 0) = -1;
    e(8, 1) = -1;
    e(8, 2) = 0;

    e(9, 0) = 1;
    e(9, 1) = -1;
    e(9, 2) = 0;

    e(10, 0) = -1;
    e(10, 1) = 1;
    e(10, 2) = 0;

    e(11, 0) = 1;
    e(11, 1) = 0;
    e(11, 2) = 1;

    e(12, 0) = -1;
    e(12, 1) = 0;
    e(12, 2) = -1;

    e(13, 0) = 1;
    e(13, 1) = 0;
    e(13, 2) = -1;

    e(14, 0) = -1;
    e(14, 1) = 0;
    e(14, 2) = 1;

    e(15, 0) = 0;
    e(15, 1) = 1;
    e(15, 2) = 1;

    e(16, 0) = 0;
    e(16, 1) = -1;
    e(16, 2) = -1;

    e(17, 0) = 0;
    e(17, 1) = 1;
    e(17, 2) = -1;

    e(18, 0) = 0;
    e(18, 1) = -1;
    e(18, 2) = 1;

    e(19, 0) = 1;
    e(19, 1) = 1;
    e(19, 2) = 1;

    e(20, 0) = -1;
    e(20, 1) = -1;
    e(20, 2) = -1;

    e(21, 0) = 1;
    e(21, 1) = -1;
    e(21, 2) = 1;

    e(22, 0) = -1;
    e(22, 1) = 1;
    e(22, 2) = -1;

    e(23, 0) = 1;
    e(23, 1) = 1;
    e(23, 2) = -1;

    e(24, 0) = -1;
    e(24, 1) = -1;
    e(24, 2) = 1;

    e(25, 0) = 1;
    e(25, 1) = -1;
    e(25, 2) = -1;

    e(26, 0) = -1;
    e(26, 1) = 1;
    e(26, 2) = 1;
    // generate random velocity
    typedef typename Kokkos::Random_XorShift64_Pool<>
        RandPoolType;
    RandPoolType rand_pool(5374857);
    Kokkos::parallel_for(
        "random", mdrange_policy3({0, 0, 0}, {lx, ly, lz}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            auto state = rand_pool.get_state();
            ran(i, j, k) = state.urand(100);
            rand_pool.free_state(state);
        });
    // macroscopic value initialization
    Kokkos::parallel_for(
        "initialize", mdrange_policy3({0, 0, 0}, {lx, ly, lz}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            ua(i, j, k) = 0;
            va(i, j, k) = 0;
            wa(i, j, k) = 0;
            p(i, j, k) = 0;
            rho(i, j, k) = rho0;
            usr(i, j, k) = (pow((x_lo + i - ghost - (glx - 1) / 8), 2) + pow((y_lo + j - ghost - (glx - 1) / 8), 2) + pow((z_lo + k - ghost - (glx - 1) / 8), 2) <= 16) ? 0 : 1;
            ua(i, j, k) = (u0 * 4.0 * (z_lo + k - ghost) * (glz - 1 - (z_lo + k - ghost)) / pow((glz - 1), 2)) * usr(i, j, k) * (1.0 + ran(i, j, k) * 0.001);
            va(i, j, k) = 0.0;
            wa(i, j, k) = 0.0;
        });

    // distribution function initialization
    Kokkos::parallel_for(
        "initf", mdrange_policy4({0, 0, 0, 0}, {q, lx, ly, lz}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            f(ii, i, j, k) = t(ii) * p(i, j, k) * 3.0 +
                             t(ii) * (3.0 * (e(ii, 0) * ua(i, j, k) + e(ii, 1) * va(i, j, k) + e(ii, 2) * wa(i, j, k)) +
                                      4.5 * pow((e(ii, 0) * ua(i, j, k) + e(ii, 1) * va(i, j, k) + e(ii, 2) * wa(i, j, k)), 2) -
                                      1.5 * (pow(ua(i, j, k), 2) + pow(va(i, j, k), 2) + pow(wa(i, j, k), 2)));

            ft(ii, i, j, k) = 0;
        });
};
void LBM::Collision()
{

    // collision

    Kokkos::parallel_for(
        "collision", mdrange_policy4({0, ghost, ghost, ghost}, {q, lx - ghost, ly - ghost, lz - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            double feq = t(ii) * p(i, j, k) * 3.0 +
                         t(ii) * (3.0 * (e(ii, 0) * ua(i, j, k) + e(ii, 1) * va(i, j, k) + e(ii, 2) * wa(i, j, k)) +
                                  4.5 * pow((e(ii, 0) * ua(i, j, k) + e(ii, 1) * va(i, j, k) + e(ii, 2) * wa(i, j, k)), 2) -
                                  1.5 * (pow(ua(i, j, k), 2) + pow(va(i, j, k), 2) + pow(wa(i, j, k), 2)));
            f(ii, i, j, k) -= (f(ii, i, j, k) - feq) / (tau0 + 0.5);
        });
};

void LBM::setup_subdomain()
{

    // prepare the value needs to be transfered
    // 6 faces

    m_left = buffer_t("m_left", q, ey, ez);

    m_right = buffer_t("m_right", q, ey, ez);

    m_down = buffer_t("m_down", q, ex, ey);

    m_up = buffer_t("m_up", q, ex, ey);

    m_front = buffer_t("m_front", q, ex, ez);

    m_back = buffer_t("m_back", q, ex, ez);
    // 12 lines

    m_leftup = buffer_ut("m_leftup", q, ey);

    m_rightup = buffer_ut("m_rightup", q, ey);

    m_leftdown = buffer_ut("m_leftdown", q, ey);

    m_rightdown = buffer_ut("m_rightdown", q, ey);

    m_backleft = buffer_ut("m_backleft", q, ez);

    m_backright = buffer_ut("m_backright", q, ez);

    m_frontleft = buffer_ut("m_frontleft", q, ez);

    m_frontright = buffer_ut("m_frontdown", q, ez);

    m_backdown = buffer_ut("m_backdown", q, ex);

    m_backup = buffer_ut("m_backup", q, ex);

    m_frontdown = buffer_ut("m_frontdown", q, ex);

    m_frontup = buffer_ut("m_frontup", q, ex);

    m_frontleftdown = buffer_st("m_fld", q);

    m_frontrightdown = buffer_st("m_frd", q);

    m_frontleftup = buffer_st("m_flu", q);

    m_frontrightup = buffer_st("m_fru", q);

    m_backleftdown = buffer_st("m_bld", q);

    m_backrightdown = buffer_st("m_brd", q);

    m_backleftup = buffer_st("m_blu", q);

    m_backrightup = buffer_st("m_bru", q);

    // outdirection
    // 6 faces

    m_leftout = buffer_t("m_leftout", q, ey, ez);

    m_rightout = buffer_t("m_rightout", q, ey, ez);

    m_downout = buffer_t("m_downout", q, ex, ey);

    m_upout = buffer_t("m_upout", q, ex, ey);

    m_frontout = buffer_t("m_downout", q, ex, ez);

    m_backout = buffer_t("m_backout", q, ex, ez);

    m_leftupout = buffer_ut("m_leftupout", q, ey);

    m_rightupout = buffer_ut("m_rightupout", q, ey);

    m_leftdownout = buffer_ut("m_leftdownout", q, ey);

    m_rightdownout = buffer_ut("m_rightdownout", q, ey);

    m_backleftout = buffer_ut("m_backleftout", q, ez);

    m_backrightout = buffer_ut("m_backrightout", q, ez);

    m_frontleftout = buffer_ut("m_frontleftout", q, ez);

    m_frontrightout = buffer_ut("m_frontdownout", q, ez);

    m_backdownout = buffer_ut("m_backdownout", q, ex);

    m_backupout = buffer_ut("m_backupout", q, ex);

    m_frontdownout = buffer_ut("m_frontdownout", q, ex);

    m_frontupout = buffer_ut("m_frontupout", q, ex);

    m_frontleftdownout = buffer_st("m_fldout", q);

    m_frontrightdownout = buffer_st("m_frdout", q);

    m_frontleftupout = buffer_st("m_fluout", q);

    m_frontrightupout = buffer_st("m_fruout", q);

    m_backleftdownout = buffer_st("m_bldout", q);

    m_backrightdownout = buffer_st("m_brdout", q);

    m_backleftupout = buffer_st("m_bluout", q);

    m_backrightupout = buffer_st("m_bruout", q);
}
void LBM::pack()
{
    // 6 faces

    Kokkos::deep_copy(m_leftout, Kokkos::subview(f, Kokkos::ALL, ghost, std::make_pair(ghost, ly - ghost), std::make_pair(ghost, lz - ghost)));

    Kokkos::deep_copy(m_rightout, Kokkos::subview(f, Kokkos::ALL, lx - ghost - 1, std::make_pair(ghost, ly - ghost), std::make_pair(ghost, lz - ghost)));

    Kokkos::deep_copy(m_downout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), std::make_pair(ghost, ly - ghost), ghost));

    Kokkos::deep_copy(m_upout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), std::make_pair(ghost, ly - ghost), lz - ghost - 1));

    Kokkos::deep_copy(m_frontout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ghost, std::make_pair(ghost, lz - ghost)));

    Kokkos::deep_copy(m_backout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ly - ghost - 1, std::make_pair(ghost, lz - ghost)));
    // 12 lines

    Kokkos::deep_copy(m_leftupout, Kokkos::subview(f, Kokkos::ALL, ghost, std::make_pair(ghost, ly - ghost), lz - ghost - 1));

    Kokkos::deep_copy(m_rightupout, Kokkos::subview(f, Kokkos::ALL, lx - ghost - 1, std::make_pair(ghost, ly - ghost), lz - ghost - 1));

    Kokkos::deep_copy(m_frontleftout, Kokkos::subview(f, Kokkos::ALL, ghost, ghost, std::make_pair(ghost, lz - ghost)));

    Kokkos::deep_copy(m_frontrightout, Kokkos::subview(f, Kokkos::ALL, lx - ghost - 1, ghost, std::make_pair(ghost, lz - ghost)));

    Kokkos::deep_copy(m_leftdownout, Kokkos::subview(f, Kokkos::ALL, ghost, std::make_pair(ghost, ly - ghost), ghost));

    Kokkos::deep_copy(m_rightdownout, Kokkos::subview(f, Kokkos::ALL, lx - ghost - 1, std::make_pair(ghost, ly - ghost), ghost));

    Kokkos::deep_copy(m_backleftout, Kokkos::subview(f, Kokkos::ALL, ghost, ly - ghost - 1, std::make_pair(ghost, lz - ghost)));

    Kokkos::deep_copy(m_backrightout, Kokkos::subview(f, Kokkos::ALL, lx - ghost - 1, ly - ghost - 1, std::make_pair(ghost, lz - ghost)));

    Kokkos::deep_copy(m_frontupout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ghost, lz - ghost - 1));

    Kokkos::deep_copy(m_frontdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ghost, ghost));

    Kokkos::deep_copy(m_backupout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ly - ghost - 1, lz - ghost - 1));

    Kokkos::deep_copy(m_backdownout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ly - ghost - 1, ghost));
    // 8 points

    Kokkos::deep_copy(m_frontleftdownout, Kokkos::subview(f, Kokkos::ALL, ghost, ghost, ghost));

    Kokkos::deep_copy(m_frontrightdownout, Kokkos::subview(f, Kokkos::ALL, lx - ghost - 1, ghost, ghost));

    Kokkos::deep_copy(m_backleftdownout, Kokkos::subview(f, Kokkos::ALL, ghost, ly - ghost - 1, ghost));

    Kokkos::deep_copy(m_backrightdownout, Kokkos::subview(f, Kokkos::ALL, lx - ghost - 1, ly - ghost - 1, ghost));

    Kokkos::deep_copy(m_frontleftupout, Kokkos::subview(f, Kokkos::ALL, ghost, ghost, lz - ghost - 1));

    Kokkos::deep_copy(m_frontrightupout, Kokkos::subview(f, Kokkos::ALL, lx - ghost - 1, ghost, lz - ghost - 1));

    Kokkos::deep_copy(m_backleftupout, Kokkos::subview(f, Kokkos::ALL, ghost, ly - ghost - 1, lz - ghost - 1));

    Kokkos::deep_copy(m_backrightupout, Kokkos::subview(f, Kokkos::ALL, lx - ghost - 1, ly - ghost - 1, lz - ghost - 1));
}

void LBM::exchange()
{
    // 6 faces
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

    if (z_lo != 0)
        MPI_Send(m_downout.data(), m_downout.size(), MPI_DOUBLE, comm.down, mar, comm.comm);

    if (z_hi != glz)
        MPI_Recv(m_up.data(), m_up.size(), MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 4;
    if (z_hi != glz)
        MPI_Send(m_upout.data(), m_upout.size(), MPI_DOUBLE, comm.up, mar, comm.comm);

    if (z_lo != 0)
        MPI_Recv(m_down.data(), m_down.size(), MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 5;
    if (y_lo != 0)
        MPI_Send(m_frontout.data(), m_frontout.size(), MPI_DOUBLE, comm.front, mar, comm.comm);

    if (y_hi != gly)
        MPI_Recv(m_back.data(), m_back.size(), MPI_DOUBLE, comm.back, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 6;
    if (y_hi != gly)
        MPI_Send(m_backout.data(), m_backout.size(), MPI_DOUBLE, comm.back, mar, comm.comm);

    if (y_lo != 0)
        MPI_Recv(m_front.data(), m_front.size(), MPI_DOUBLE, comm.front, mar, comm.comm, MPI_STATUSES_IGNORE);
    // 12 lines
    mar = 7;
    if (x_lo != 0 && y_lo != 0)
        MPI_Send(m_frontleftout.data(), m_frontleftout.size(), MPI_DOUBLE, comm.frontleft, mar, comm.comm);

    if (x_hi != glx && y_hi != gly)
        MPI_Recv(m_backright.data(), m_backright.size(), MPI_DOUBLE, comm.backright, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 8;
    if (x_hi != glx && y_hi != gly)
        MPI_Send(m_backrightout.data(), m_backrightout.size(), MPI_DOUBLE, comm.backright, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0)
        MPI_Recv(m_frontleft.data(), m_frontleft.size(), MPI_DOUBLE, comm.frontleft, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 9;
    if (x_hi != glx && y_lo != 0)
        MPI_Send(m_frontrightout.data(), m_frontrightout.size(), MPI_DOUBLE, comm.frontright, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly)
        MPI_Recv(m_backleft.data(), m_backleft.size(), MPI_DOUBLE, comm.backleft, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 10;
    if (x_lo != 0 && y_hi != gly)
        MPI_Send(m_backleftout.data(), m_backleftout.size(), MPI_DOUBLE, comm.backleft, mar, comm.comm);

    if (x_hi != glx && y_lo != 0)
        MPI_Recv(m_frontright.data(), m_frontright.size(), MPI_DOUBLE, comm.frontright, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 11;
    if (x_lo != 0 && z_hi != glz)
        MPI_Send(m_leftupout.data(), m_leftupout.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm);

    if (x_hi != glx && z_lo != 0)
        MPI_Recv(m_rightdown.data(), m_rightdown.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 12;

    if (x_hi != glx && z_lo != 0)
        MPI_Send(m_rightdownout.data(), m_rightdownout.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm);

    if (x_lo != 0 && z_hi != glz)
        MPI_Recv(m_leftup.data(), m_leftup.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 13;
    if (x_lo != 0 && z_lo != 0)
        MPI_Send(m_leftdownout.data(), m_leftdownout.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm);

    if (x_hi != glx && z_hi != glz)
        MPI_Recv(m_rightup.data(), m_rightup.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 14;
    if (x_hi != glx && z_hi != glz)
        MPI_Send(m_rightupout.data(), m_rightupout.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm);

    if (x_lo != 0 && z_lo != 0)
        MPI_Recv(m_leftdown.data(), m_leftdown.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 15;
    if (y_lo != 0 && z_hi != glz)
        MPI_Send(m_frontupout.data(), m_frontupout.size(), MPI_DOUBLE, comm.frontup, mar, comm.comm);

    if (y_hi != gly && z_lo != 0)
        MPI_Recv(m_backdown.data(), m_backdown.size(), MPI_DOUBLE, comm.backdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 16;
    if (y_hi != gly && z_lo != 0)
        MPI_Send(m_backdownout.data(), m_backdownout.size(), MPI_DOUBLE, comm.backdown, mar, comm.comm);

    if (y_lo != 0 && z_hi != glz)
        MPI_Recv(m_frontup.data(), m_frontup.size(), MPI_DOUBLE, comm.frontup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 17;
    if (y_lo != 0 && z_lo != 0)
        MPI_Send(m_frontdownout.data(), m_frontdownout.size(), MPI_DOUBLE, comm.frontdown, mar, comm.comm);

    if (y_hi != gly && z_hi != glz)
        MPI_Recv(m_backup.data(), m_backup.size(), MPI_DOUBLE, comm.backup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 18;
    if (y_hi != gly && z_hi != glz)
        MPI_Send(m_backupout.data(), m_backupout.size(), MPI_DOUBLE, comm.backup, mar, comm.comm);

    if (y_lo != 0 && z_lo != 0)
        MPI_Recv(m_frontdown.data(), m_frontdown.size(), MPI_DOUBLE, comm.frontdown, mar, comm.comm, MPI_STATUSES_IGNORE);
    // 8 points
    mar = 19;
    if (x_lo != 0 && y_lo != 0 && z_lo != 0)
        MPI_Send(m_frontleftdownout.data(), m_frontleftdownout.size(), MPI_DOUBLE, comm.frontleftdown, mar, comm.comm);

    if (x_hi != glx && y_hi != gly && z_hi != glz)
        MPI_Recv(m_backrightup.data(), m_backrightup.size(), MPI_DOUBLE, comm.backrightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 20;
    if (x_hi != glx && y_hi != gly && z_hi != glz)
        MPI_Send(m_backrightupout.data(), m_backrightupout.size(), MPI_DOUBLE, comm.backrightup, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0 && z_lo != 0)
        MPI_Recv(m_frontleftdown.data(), m_frontleftdown.size(), MPI_DOUBLE, comm.frontleftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 21;
    if (x_lo != 0 && y_hi != gly && z_hi != glz)
        MPI_Send(m_backleftupout.data(), m_backleftupout.size(), MPI_DOUBLE, comm.backleftup, mar, comm.comm);

    if (x_hi != glx && y_lo != 0 && z_lo != 0)
        MPI_Recv(m_frontrightdown.data(), m_frontrightdown.size(), MPI_DOUBLE, comm.frontrightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 22;
    if (x_hi != glx && y_lo != 0 && z_lo != 0)
        MPI_Send(m_frontrightdownout.data(), m_frontrightdownout.size(), MPI_DOUBLE, comm.frontrightdown, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly && z_hi != glz)
        MPI_Recv(m_backleftup.data(), m_backleftup.size(), MPI_DOUBLE, comm.backleftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 23;
    if (x_lo != 0 && y_hi != gly && z_lo != 0)
        MPI_Send(m_backleftdownout.data(), m_backleftdownout.size(), MPI_DOUBLE, comm.backleftdown, mar, comm.comm);

    if (x_hi != glx && y_lo != 0 && z_hi != glz)
        MPI_Recv(m_frontrightup.data(), m_frontrightup.size(), MPI_DOUBLE, comm.frontrightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 24;
    if (x_hi != glx && y_lo != 0 && z_hi != glz)
        MPI_Send(m_frontrightupout.data(), m_frontrightupout.size(), MPI_DOUBLE, comm.frontrightup, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly && z_lo != 0)
        MPI_Recv(m_backleftdown.data(), m_backleftdown.size(), MPI_DOUBLE, comm.backleftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 25;
    if (x_lo != 0 && y_lo != 0 && z_hi != glz)
        MPI_Send(m_frontleftupout.data(), m_frontleftupout.size(), MPI_DOUBLE, comm.frontleftup, mar, comm.comm);

    if (x_hi != glx && y_hi != gly && z_lo != 0)
        MPI_Recv(m_backrightdown.data(), m_backrightdown.size(), MPI_DOUBLE, comm.backrightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 26;
    if (x_hi != glx && y_hi != gly && z_lo != 0)
        MPI_Send(m_backrightdownout.data(), m_backrightdownout.size(), MPI_DOUBLE, comm.backrightdown, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0 && z_hi != glz)
        MPI_Recv(m_frontleftup.data(), m_frontleftup.size(), MPI_DOUBLE, comm.frontleftup, mar, comm.comm, MPI_STATUSES_IGNORE);

    // edges outsides

    if (x_lo == 0)
    {

        mar = 27;
        if (z_hi != glz)
            MPI_Send(m_leftupout.data(), m_leftupout.size(), MPI_DOUBLE, comm.up, mar, comm.comm);

        if (z_lo != 0)
            MPI_Recv(m_leftdown.data(), m_leftdown.size(), MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 28;
        if (y_hi != gly)
            MPI_Send(m_backleftout.data(), m_backleftout.size(), MPI_DOUBLE, comm.back, mar, comm.comm);
        if (y_lo != 0)
            MPI_Recv(m_frontleft.data(), m_frontleft.size(), MPI_DOUBLE, comm.front, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 29;
        if (z_lo != 0)
            MPI_Send(m_leftdownout.data(), m_leftdownout.size(), MPI_DOUBLE, comm.down, mar, comm.comm);

        if (z_hi != glz)
            MPI_Recv(m_leftup.data(), m_leftup.size(), MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 30;
        if (y_lo != 0)
            MPI_Send(m_frontleftout.data(), m_frontleftout.size(), MPI_DOUBLE, comm.front, mar, comm.comm);
        if (y_hi != gly)
            MPI_Recv(m_backleft.data(), m_backleft.size(), MPI_DOUBLE, comm.back, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 51;
        if (z_hi != glz && y_lo != 0)
            MPI_Send(m_frontleftupout.data(), m_frontleftupout.size(), MPI_DOUBLE, comm.frontup, mar, comm.comm);

        if (z_lo != 0 && y_hi != gly)
            MPI_Recv(m_backleftdown.data(), m_backleftdown.size(), MPI_DOUBLE, comm.backdown, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 52;
        if (y_hi != gly && z_hi != glz)
            MPI_Send(m_backleftupout.data(), m_backleftupout.size(), MPI_DOUBLE, comm.backup, mar, comm.comm);
        if (y_lo != 0 && z_lo != 0)
            MPI_Recv(m_frontleftdown.data(), m_frontleftdown.size(), MPI_DOUBLE, comm.frontdown, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 53;
        if (z_lo != 0 && y_lo != 0)
            MPI_Send(m_frontleftdownout.data(), m_frontleftdownout.size(), MPI_DOUBLE, comm.frontdown, mar, comm.comm);

        if (z_hi != glz && y_hi != gly)
            MPI_Recv(m_backleftup.data(), m_backleftup.size(), MPI_DOUBLE, comm.backup, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 54;
        if (y_hi != gly && z_lo != 0)
            MPI_Send(m_backleftdownout.data(), m_backleftdownout.size(), MPI_DOUBLE, comm.backdown, mar, comm.comm);
        if (y_lo != 0 && z_hi != glz)
            MPI_Recv(m_frontleftup.data(), m_frontleftup.size(), MPI_DOUBLE, comm.frontup, mar, comm.comm, MPI_STATUSES_IGNORE);
    }

    if (x_hi == glx)
    {

        mar = 31;
        if (z_hi != glz)
            MPI_Send(m_rightupout.data(), m_rightupout.size(), MPI_DOUBLE, comm.up, mar, comm.comm);

        if (z_lo != 0)
            MPI_Recv(m_rightdown.data(), m_rightdown.size(), MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 32;
        if (y_hi != gly)
            MPI_Send(m_backrightout.data(), m_backrightout.size(), MPI_DOUBLE, comm.back, mar, comm.comm);
        if (y_lo != 0)
            MPI_Recv(m_frontright.data(), m_frontright.size(), MPI_DOUBLE, comm.front, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 33;
        if (z_lo != 0)
            MPI_Send(m_rightdownout.data(), m_rightdownout.size(), MPI_DOUBLE, comm.down, mar, comm.comm);

        if (z_hi != glz)
            MPI_Recv(m_rightup.data(), m_rightup.size(), MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 34;
        if (y_lo != 0)
            MPI_Send(m_frontrightout.data(), m_frontrightout.size(), MPI_DOUBLE, comm.front, mar, comm.comm);
        if (y_hi != gly)
            MPI_Recv(m_backright.data(), m_backright.size(), MPI_DOUBLE, comm.back, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 55;
        if (z_hi != glz && y_lo != 0)
            MPI_Send(m_frontrightupout.data(), m_frontrightupout.size(), MPI_DOUBLE, comm.frontup, mar, comm.comm);

        if (z_lo != 0 && y_hi != gly)
            MPI_Recv(m_backrightdown.data(), m_backrightdown.size(), MPI_DOUBLE, comm.backdown, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 56;
        if (y_hi != gly && z_hi != glz)
            MPI_Send(m_backrightupout.data(), m_backrightupout.size(), MPI_DOUBLE, comm.backup, mar, comm.comm);
        if (y_lo != 0 && z_lo != 0)
            MPI_Recv(m_frontrightdown.data(), m_frontrightdown.size(), MPI_DOUBLE, comm.frontdown, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 57;
        if (z_lo != 0 && y_lo != 0)
            MPI_Send(m_frontrightdownout.data(), m_frontrightdownout.size(), MPI_DOUBLE, comm.frontdown, mar, comm.comm);

        if (z_hi != glz && y_hi != gly)
            MPI_Recv(m_backrightup.data(), m_backrightup.size(), MPI_DOUBLE, comm.backup, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 58;
        if (y_hi != gly && z_lo != 0)
            MPI_Send(m_backrightdownout.data(), m_backrightdownout.size(), MPI_DOUBLE, comm.backdown, mar, comm.comm);
        if (y_lo != 0 && z_hi != glz)
            MPI_Recv(m_frontrightup.data(), m_frontrightup.size(), MPI_DOUBLE, comm.frontup, mar, comm.comm, MPI_STATUSES_IGNORE);
    }

    if (y_lo == 0)
    {

        mar = 35;
        if (z_hi != glz)
            MPI_Send(m_frontupout.data(), m_frontupout.size(), MPI_DOUBLE, comm.up, mar, comm.comm);

        if (z_lo != 0)
            MPI_Recv(m_frontdown.data(), m_frontdown.size(), MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 36;
        if (x_hi != glx)
            MPI_Send(m_frontrightout.data(), m_frontrightout.size(), MPI_DOUBLE, comm.right, mar, comm.comm);
        if (x_lo != 0)
            MPI_Recv(m_frontleft.data(), m_frontleft.size(), MPI_DOUBLE, comm.left, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 37;
        if (z_lo != 0)
            MPI_Send(m_frontdownout.data(), m_frontdownout.size(), MPI_DOUBLE, comm.down, mar, comm.comm);

        if (z_hi != glz)
            MPI_Recv(m_frontup.data(), m_frontup.size(), MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 38;
        if (x_lo != 0)
            MPI_Send(m_frontleftout.data(), m_frontleftout.size(), MPI_DOUBLE, comm.left, mar, comm.comm);
        if (x_hi != glx)
            MPI_Recv(m_frontright.data(), m_frontright.size(), MPI_DOUBLE, comm.right, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 59;
        if (z_hi != glz && x_lo != 0)
            MPI_Send(m_frontleftupout.data(), m_frontleftupout.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm);

        if (z_lo != 0 && x_hi != glx)
            MPI_Recv(m_frontrightdown.data(), m_frontrightdown.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 60;
        if (x_hi != glx && z_hi != glz)
            MPI_Send(m_frontrightupout.data(), m_frontrightupout.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm);
        if (x_lo != 0 && z_lo != 0)
            MPI_Recv(m_frontleftdown.data(), m_frontleftdown.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 61;
        if (z_lo != 0 && x_lo != 0)
            MPI_Send(m_frontleftdownout.data(), m_frontleftdownout.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm);

        if (z_hi != glz && x_hi != glx)
            MPI_Recv(m_frontrightup.data(), m_frontrightup.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 62;
        if (x_hi != glx && z_lo != 0)
            MPI_Send(m_frontrightdownout.data(), m_frontrightdownout.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm);
        if (x_lo != 0 && z_hi != glz)
            MPI_Recv(m_frontleftup.data(), m_frontleftup.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm, MPI_STATUSES_IGNORE);
    }

    if (y_hi == gly)
    {

        mar = 39;
        if (z_hi != glz)
            MPI_Send(m_backupout.data(), m_backupout.size(), MPI_DOUBLE, comm.up, mar, comm.comm);

        if (z_lo != 0)
            MPI_Recv(m_backdown.data(), m_backdown.size(), MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 40;
        if (x_hi != glx)
            MPI_Send(m_backrightout.data(), m_backrightout.size(), MPI_DOUBLE, comm.right, mar, comm.comm);
        if (x_lo != 0)
            MPI_Recv(m_backleft.data(), m_backleft.size(), MPI_DOUBLE, comm.left, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 41;
        if (z_lo != 0)
            MPI_Send(m_backdownout.data(), m_backdownout.size(), MPI_DOUBLE, comm.down, mar, comm.comm);

        if (z_hi != glz)
            MPI_Recv(m_backup.data(), m_backup.size(), MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 42;
        if (x_lo != 0)
            MPI_Send(m_backleftout.data(), m_backleftout.size(), MPI_DOUBLE, comm.left, mar, comm.comm);
        if (x_hi != glx)
            MPI_Recv(m_backright.data(), m_backright.size(), MPI_DOUBLE, comm.right, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 63;
        if (z_hi != glz && x_lo != 0)
            MPI_Send(m_backleftupout.data(), m_backleftupout.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm);

        if (z_lo != 0 && x_hi != glx)
            MPI_Recv(m_backrightdown.data(), m_backrightdown.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 64;
        if (x_hi != glx && z_hi != glz)
            MPI_Send(m_backrightupout.data(), m_backrightupout.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm);
        if (x_lo != 0 && z_lo != 0)
            MPI_Recv(m_backleftdown.data(), m_backleftdown.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 65;
        if (z_lo != 0 && x_lo != 0)
            MPI_Send(m_backleftdownout.data(), m_backleftdownout.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm);

        if (z_hi != glz && x_hi != glx)
            MPI_Recv(m_backrightup.data(), m_backrightup.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 66;
        if (x_hi != glx && z_lo != 0)
            MPI_Send(m_backrightdownout.data(), m_backrightdownout.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm);
        if (x_lo != 0 && z_hi != glz)
            MPI_Recv(m_backleftup.data(), m_backleftup.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm, MPI_STATUSES_IGNORE);
    }

    if (z_hi == glz)
    {

        mar = 43;
        if (y_hi != gly)
            MPI_Send(m_backupout.data(), m_backupout.size(), MPI_DOUBLE, comm.back, mar, comm.comm);

        if (y_lo != 0)
            MPI_Recv(m_frontup.data(), m_frontup.size(), MPI_DOUBLE, comm.front, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 44;
        if (x_hi != glx)
            MPI_Send(m_rightupout.data(), m_rightupout.size(), MPI_DOUBLE, comm.right, mar, comm.comm);
        if (x_lo != 0)
            MPI_Recv(m_leftup.data(), m_leftup.size(), MPI_DOUBLE, comm.left, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 45;
        if (y_lo != 0)
            MPI_Send(m_frontupout.data(), m_frontupout.size(), MPI_DOUBLE, comm.front, mar, comm.comm);

        if (y_hi != gly)
            MPI_Recv(m_backup.data(), m_backup.size(), MPI_DOUBLE, comm.back, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 46;
        if (x_lo != 0)
            MPI_Send(m_leftupout.data(), m_leftupout.size(), MPI_DOUBLE, comm.left, mar, comm.comm);
        if (x_hi != glx)
            MPI_Recv(m_rightup.data(), m_rightup.size(), MPI_DOUBLE, comm.right, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 67;
        if (y_hi != gly && x_lo != 0)
            MPI_Send(m_backleftupout.data(), m_backleftupout.size(), MPI_DOUBLE, comm.backleft, mar, comm.comm);

        if (y_lo != 0 && x_hi != glx)
            MPI_Recv(m_frontrightup.data(), m_frontrightup.size(), MPI_DOUBLE, comm.frontright, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 68;
        if (x_hi != glx && y_hi != gly)
            MPI_Send(m_backrightupout.data(), m_backrightupout.size(), MPI_DOUBLE, comm.backright, mar, comm.comm);
        if (x_lo != 0 && y_lo != 0)
            MPI_Recv(m_frontleftup.data(), m_frontleftup.size(), MPI_DOUBLE, comm.frontleft, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 69;
        if (y_lo != 0 && x_lo != 0)
            MPI_Send(m_frontleftupout.data(), m_frontleftupout.size(), MPI_DOUBLE, comm.frontleft, mar, comm.comm);

        if (y_hi != gly && x_hi != glx)
            MPI_Recv(m_backrightup.data(), m_backrightup.size(), MPI_DOUBLE, comm.backright, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 70;
        if (x_hi != glx && y_lo != 0)
            MPI_Send(m_frontrightupout.data(), m_frontrightupout.size(), MPI_DOUBLE, comm.frontright, mar, comm.comm);
        if (x_lo != 0 && y_hi != gly)
            MPI_Recv(m_backleftup.data(), m_backleftup.size(), MPI_DOUBLE, comm.backleft, mar, comm.comm, MPI_STATUSES_IGNORE);
    }

    if (z_lo == 0)
    {

        mar = 47;
        if (y_hi != gly)
            MPI_Send(m_backdownout.data(), m_backdownout.size(), MPI_DOUBLE, comm.back, mar, comm.comm);

        if (y_lo != 0)
            MPI_Recv(m_frontdown.data(), m_frontdown.size(), MPI_DOUBLE, comm.front, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 48;
        if (x_hi != glx)
            MPI_Send(m_rightdownout.data(), m_rightdownout.size(), MPI_DOUBLE, comm.right, mar, comm.comm);
        if (x_lo != 0)
            MPI_Recv(m_leftdown.data(), m_leftdown.size(), MPI_DOUBLE, comm.left, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 49;
        if (y_lo != 0)
            MPI_Send(m_frontdownout.data(), m_frontdownout.size(), MPI_DOUBLE, comm.front, mar, comm.comm);

        if (y_hi != gly)
            MPI_Recv(m_backdown.data(), m_backdown.size(), MPI_DOUBLE, comm.back, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 50;
        if (x_lo != 0)
            MPI_Send(m_leftdownout.data(), m_leftdownout.size(), MPI_DOUBLE, comm.left, mar, comm.comm);
        if (x_hi != glx)
            MPI_Recv(m_rightdown.data(), m_rightdown.size(), MPI_DOUBLE, comm.right, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 71;
        if (y_hi != gly && x_lo != 0)
            MPI_Send(m_backleftdownout.data(), m_backleftdownout.size(), MPI_DOUBLE, comm.backleft, mar, comm.comm);

        if (y_lo != 0 && x_hi != glx)
            MPI_Recv(m_frontrightdown.data(), m_frontrightdown.size(), MPI_DOUBLE, comm.frontright, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 72;
        if (x_hi != glx && y_hi != gly)
            MPI_Send(m_backrightdownout.data(), m_backrightdownout.size(), MPI_DOUBLE, comm.backright, mar, comm.comm);
        if (x_lo != 0 && y_lo != 0)
            MPI_Recv(m_frontleftdown.data(), m_frontleftdown.size(), MPI_DOUBLE, comm.frontleft, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 73;
        if (y_lo != 0 && x_lo != 0)
            MPI_Send(m_frontleftdownout.data(), m_frontleftdownout.size(), MPI_DOUBLE, comm.frontleft, mar, comm.comm);

        if (y_hi != gly && x_hi != glx)
            MPI_Recv(m_backrightdown.data(), m_backrightdown.size(), MPI_DOUBLE, comm.backright, mar, comm.comm, MPI_STATUSES_IGNORE);

        mar = 74;
        if (x_hi != glx && y_lo != 0)
            MPI_Send(m_frontrightdownout.data(), m_frontrightdownout.size(), MPI_DOUBLE, comm.frontright, mar, comm.comm);
        if (x_lo != 0 && y_hi != gly)
            MPI_Recv(m_backleftdown.data(), m_backleftdown.size(), MPI_DOUBLE, comm.backleft, mar, comm.comm, MPI_STATUSES_IGNORE);
    }
}

void LBM::unpack()
{
    // 6 faces
    if (x_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, std::make_pair(ghost, ly - ghost), std::make_pair(ghost, lz - ghost)), m_left);

    if (x_hi != glx)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, std::make_pair(ghost, ly - ghost), std::make_pair(ghost, lz - ghost)), m_right);

    if (z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), std::make_pair(ghost, ly - ghost), ghost - 1), m_down);

    if (z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), std::make_pair(ghost, ly - ghost), lz - ghost), m_up);

    if (y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ghost - 1, std::make_pair(ghost, lz - ghost)), m_front);

    if (y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ly - ghost, std::make_pair(ghost, lz - ghost)), m_back);
    // 12 lines
    if (x_lo != 0 && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, std::make_pair(ghost, ly - ghost), lz - ghost), m_leftup);

    if (x_hi != glx && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, std::make_pair(ghost, ly - ghost), lz - ghost), m_rightup);

    if (x_lo != 0 && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ghost - 1, std::make_pair(ghost, lz - ghost)), m_frontleft);

    if (x_hi != glx && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ghost - 1, std::make_pair(ghost, lz - ghost)), m_frontright);

    if (x_lo != 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, std::make_pair(ghost, ly - ghost), ghost - 1), m_leftdown);

    if (x_hi != glx && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, std::make_pair(ghost, ly - ghost), ghost - 1), m_rightdown);

    if (x_lo != 0 && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ly - ghost, std::make_pair(ghost, lz - ghost)), m_backleft);

    if (x_hi != glx && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ly - ghost, std::make_pair(ghost, lz - ghost)), m_backright);

    if (y_lo != 0 && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ghost - 1, lz - ghost), m_frontup);

    if (y_lo != 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ghost - 1, ghost - 1), m_frontdown);

    if (y_hi != gly && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ly - ghost, lz - ghost), m_backup);

    if (y_hi != gly && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ly - ghost, ghost - 1), m_backdown);
    // 8 points
    if (x_lo != 0 && y_lo != 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ghost - 1, ghost - 1), m_frontleftdown);
    if (x_hi != glx && y_lo != 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ghost - 1, ghost - 1), m_frontrightdown);
    if (x_lo != 0 && y_hi != gly && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ly - ghost, ghost - 1), m_backleftdown);
    if (x_hi != glx && y_hi != gly && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ly - ghost, ghost - 1), m_backrightdown);
    if (x_lo != 0 && y_lo != 0 && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ghost - 1, lz - ghost), m_frontleftup);
    if (x_hi != glx && y_lo != 0 && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ghost - 1, lz - ghost), m_frontrightup);
    if (x_lo != 0 && y_hi != gly && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ly - ghost, lz - ghost), m_backleftup);
    if (x_hi != glx && y_hi != gly && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ly - ghost, lz - ghost), m_backrightup);

    if (x_lo == 0 && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, std::make_pair(ghost, ly - ghost), lz - ghost), m_leftup);

    if (x_lo == 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, std::make_pair(ghost, ly - ghost), ghost - 1), m_leftdown);

    if (x_lo == 0 && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ly - ghost, std::make_pair(ghost, lz - ghost)), m_backleft);

    if (x_lo == 0 && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ghost - 1, std::make_pair(ghost, lz - ghost)), m_frontleft);

    if (x_hi == glx && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, std::make_pair(ghost, ly - ghost), lz - ghost), m_rightup);

    if (x_hi == glx && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, std::make_pair(ghost, ly - ghost), ghost - 1), m_rightdown);

    if (x_hi == glx && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ly - ghost, std::make_pair(ghost, lz - ghost)), m_backright);

    if (x_hi == glx && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ghost - 1, std::make_pair(ghost, lz - ghost)), m_frontright);

    if (y_lo == 0 && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ghost - 1, lz - ghost), m_frontup);

    if (y_lo == 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ghost - 1, ghost - 1), m_frontdown);

    if (y_lo == 0 && x_hi != glx)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ghost - 1, std::make_pair(ghost, lz - ghost)), m_frontright);

    if (y_lo == 0 && x_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ghost - 1, std::make_pair(ghost, lz - ghost)), m_frontleft);

    if (y_hi == gly && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ly - ghost, lz - ghost), m_backup);

    if (y_hi == gly && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ly - ghost, ghost - 1), m_backdown);

    if (y_hi == gly && x_hi != glx)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ly - ghost, std::make_pair(ghost, lz - ghost)), m_backright);

    if (y_hi == gly && x_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ly - ghost, std::make_pair(ghost, lz - ghost)), m_backleft);

    if (z_lo == 0 && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ly - ghost, ghost - 1), m_backdown);

    if (z_lo == 0 && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ghost - 1, ghost - 1), m_backdown);

    if (z_lo == 0 && x_hi != glx)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, std::make_pair(ghost, ly - ghost), ghost - 1), m_rightdown);

    if (z_lo == 0 && x_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, std::make_pair(ghost, ly - ghost), ghost - 1), m_leftdown);

    if (z_hi == glz && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ly - ghost, lz - ghost), m_backup);

    if (z_hi == glz && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(ghost, lx - ghost), ghost - 1, lz - ghost), m_frontup);

    if (z_hi == glz && x_hi != glx)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, std::make_pair(ghost, ly - ghost), lz - ghost), m_rightup);

    if (z_hi == glz && x_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, std::make_pair(ghost, ly - ghost), lz - ghost), m_leftup);

    if (x_lo == 0 && y_lo != 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ghost - 1, ghost - 1), m_frontleftdown);
    if (x_lo == 0 && y_lo != 0 && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ghost - 1, ghost - 1), m_frontleftup);
    if (x_lo == 0 && y_hi != gly && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ly - ghost, ghost - 1), m_backleftdown);
    if (x_lo == 0 && y_hi != gly && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ly - ghost, ghost - 1), m_backleftup);

    if (x_hi == glx && y_lo != 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ghost - 1, lz - ghost), m_frontrightup);
    if (x_hi == glx && y_lo != 0 && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ghost - 1, ghost - 1), m_frontrightdown);
    if (x_hi == glx && y_hi != gly && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ly - ghost, lz - ghost), m_backrightup);
    if (x_hi == glx && y_hi != gly && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ly - ghost, ghost - 1), m_backrightdown);

    if (y_lo == 0 && x_lo != 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ghost - 1, ghost - 1), m_frontleftdown);
    if (y_lo == 0 && x_lo != 0 && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ghost - 1, lz - ghost), m_frontleftup);
    if (y_lo == 0 && x_hi != glx && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ghost - 1, lz - ghost), m_frontrightup);
    if (y_lo == 0 && x_hi != glx && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ghost - 1, ghost - 1), m_frontrightdown);

    if (y_hi == gly && x_lo != 0 && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ly - ghost, ghost - 1), m_backleftdown);
    if (y_hi == gly && x_lo != 0 && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ly - ghost, lz - ghost), m_backleftup);
    if (y_hi == gly && x_hi != glx && z_hi != glz)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ly - ghost, lz - ghost), m_backrightup);
    if (y_hi == gly && x_hi != glx && z_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ly - ghost, ghost - 1), m_backrightdown);

    if (z_lo == 0 && x_lo != 0 && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ghost - 1, ghost - 1), m_frontleftdown);
    if (z_lo == 0 && x_lo != 0 && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ghost - 1, lz - ghost), m_backleftdown);
    if (z_lo == 0 && x_hi != glx && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ghost - 1, lz - ghost), m_backrightdown);
    if (z_lo == 0 && x_hi != glx && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ghost - 1, ghost - 1), m_frontrightdown);

    if (z_hi == glz && x_lo != 0 && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ly - ghost, ghost - 1), m_frontleftup);
    if (z_hi == glz && x_lo != 0 && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, ghost - 1, ly - ghost, lz - ghost), m_backleftup);
    if (z_hi == glz && x_hi != glx && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ly - ghost, lz - ghost), m_backrightup);
    if (z_hi == glz && x_hi != glx && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - ghost, ly - ghost, ghost - 1), m_frontrightup);
}

void LBM::Streaming()
{
    // left boundary free flow
    if (x_lo == 0)
    {
        Kokkos::parallel_for(
            "bcl", mdrange_policy2({ghost - 1, ghost - 1}, {ly - ghost + 1, lz - ghost + 1}), KOKKOS_CLASS_LAMBDA(const int j, const int k) {
                f(1, ghost - 1, j, k) = f(1, ghost, j, k);
                f(7, ghost - 1, j, k) = f(7, ghost, j, k);
                f(9, ghost - 1, j, k) = f(9, ghost, j, k);
                f(11, ghost - 1, j, k) = f(11, ghost, j, k);
                f(13, ghost - 1, j, k) = f(13, ghost, j, k);
                f(19, ghost - 1, j, k) = f(19, ghost, j, k);
                f(21, ghost - 1, j, k) = f(21, ghost, j, k);
                f(23, ghost - 1, j, k) = f(23, ghost, j, k);
                f(25, ghost - 1, j, k) = f(25, ghost, j, k);
            });
    }
    // right boundary free flow
    if (x_hi == glx)
    {
        Kokkos::parallel_for(
            "bcr", mdrange_policy2({ghost - 1, ghost - 1}, {ly - ghost + 1, lz - ghost + 1}), KOKKOS_CLASS_LAMBDA(const int j, const int k) {
                f(2, lx - ghost, j, k) = f(2, lx - ghost - 1, j, k);
                f(8, lx - ghost, j, k) = f(8, lx - ghost - 1, j - e(8, 1), k - e(8, 2));
                f(10, lx - ghost, j, k) = f(10, lx - ghost - 1, j - e(10, 1), k - e(10, 2));
                f(12, lx - ghost, j, k) = f(12, lx - ghost - 1, j - e(12, 1), k - e(12, 2));
                f(14, lx - ghost, j, k) = f(14, lx - ghost - 1, j - e(14, 1), k - e(14, 2));
                f(20, lx - ghost, j, k) = f(20, lx - ghost - 1, j - e(20, 1), k - e(20, 2));
                f(22, lx - ghost, j, k) = f(22, lx - ghost - 1, j - e(22, 1), k - e(22, 2));
                f(24, lx - ghost, j, k) = f(24, lx - ghost - 1, j - e(24, 1), k - e(24, 2));
                f(26, lx - ghost, j, k) = f(26, lx - ghost - 1, j - e(26, 1), k - e(26, 2));
            });
    }
    // front boundary bounce back
    if (y_lo == 0)
    {
        Kokkos::parallel_for(
            "bcf", mdrange_policy2({ghost - 1, ghost - 1}, {lx - ghost + 1, lz - ghost + 1}), KOKKOS_CLASS_LAMBDA(const int i, const int k) {
                f(3, i, ghost - 1, k) = f(bb(3), i, ghost + 1, k);
                f(7, i, ghost - 1, k) = f(bb(7), i + 2 * e(7, 0), ghost + 1, k + 2 * e(7, 2));
                f(10, i, ghost - 1, k) = f(bb(10), i + 2 * e(10, 0), ghost + 1, k + 2 * e(10, 2));
                f(15, i, ghost - 1, k) = f(bb(15), i + 2 * e(15, 0), ghost + 1, k + 2 * e(15, 2));
                f(17, i, ghost - 1, k) = f(bb(17), i + 2 * e(17, 0), ghost + 1, k + 2 * e(17, 2));
                f(19, i, ghost - 1, k) = f(bb(19), i + 2 * e(19, 0), ghost + 1, k + 2 * e(19, 2));
                f(22, i, ghost - 1, k) = f(bb(22), i + 2 * e(22, 0), ghost + 1, k + 2 * e(22, 2));
                f(23, i, ghost - 1, k) = f(bb(23), i + 2 * e(23, 0), ghost + 1, k + 2 * e(23, 2));
                f(26, i, ghost - 1, k) = f(bb(26), i + 2 * e(26, 0), ghost + 1, k + 2 * e(26, 2));
            });
    }
    // back boundary bounce back
    if (y_hi == gly)
    {
        Kokkos::parallel_for(
            "bcb", mdrange_policy2({ghost - 1, ghost - 1}, {lx - ghost + 1, lz - ghost + 1}), KOKKOS_CLASS_LAMBDA(const int i, const int k) {
                f(4, i, ly - ghost, k) = f(bb(4), i, ly - ghost - 2, k);
                f(8, i, ly - ghost, k) = f(bb(8), i + 2 * e(8, 0), ly - ghost - 2, k + 2 * e(8, 2));
                f(9, i, ly - ghost, k) = f(bb(9), i + 2 * e(9, 0), ly - ghost - 2, k + 2 * e(9, 2));
                f(16, i, ly - ghost, k) = f(bb(16), i + 2 * e(16, 0), ly - ghost - 2, k + 2 * e(16, 2));
                f(18, i, ly - ghost, k) = f(bb(18), i + 2 * e(18, 0), ly - ghost - 2, k + 2 * e(18, 2));
                f(20, i, ly - ghost, k) = f(bb(20), i + 2 * e(20, 0), ly - ghost - 2, k + 2 * e(20, 2));
                f(21, i, ly - ghost, k) = f(bb(21), i + 2 * e(21, 0), ly - ghost - 2, k + 2 * e(21, 2));
                f(24, i, ly - ghost, k) = f(bb(24), i + 2 * e(24, 0), ly - ghost - 2, k + 2 * e(24, 2));
                f(25, i, ly - ghost, k) = f(bb(25), i + 2 * e(25, 0), ly - ghost - 2, k + 2 * e(25, 2));
            });
    }
    // bottom boundary bounce back
    if (z_lo == 0)
    {
        Kokkos::parallel_for(
            "bcd", mdrange_policy2({ghost - 1, ghost - 1}, {lx - ghost + 1, ly - ghost + 1}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                f(5, i, j, ghost - 1) = f(bb(5), i, j, ghost + 1);
                f(11, i, j, ghost - 1) = f(bb(11), i + 2 * e(11, 0), j + 2 * e(11, 1), ghost + 1);
                f(14, i, j, ghost - 1) = f(bb(14), i + 2 * e(14, 0), j + 2 * e(14, 1), ghost + 1);
                f(15, i, j, ghost - 1) = f(bb(15), i + 2 * e(15, 0), j + 2 * e(15, 1), ghost + 1);
                f(18, i, j, ghost - 1) = f(bb(18), i + 2 * e(18, 0), j + 2 * e(18, 1), ghost + 1);
                f(19, i, j, ghost - 1) = f(bb(19), i + 2 * e(19, 0), j + 2 * e(19, 1), ghost + 1);
                f(21, i, j, ghost - 1) = f(bb(21), i + 2 * e(21, 0), j + 2 * e(21, 1), ghost + 1);
                f(24, i, j, ghost - 1) = f(bb(24), i + 2 * e(24, 0), j + 2 * e(24, 1), ghost + 1);
                f(26, i, j, ghost - 1) = f(bb(26), i + 2 * e(26, 0), j + 2 * e(26, 1), ghost + 1);
            });
    }
    // top boundary bounce back
    if (z_hi == glz)
    {
        Kokkos::parallel_for(
            "bcu", mdrange_policy2({ghost - 1, ghost - 1}, {lx - ghost + 1, ly - ghost + 1}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                f(6, i, j, lz - ghost) = f(bb(6), i, j, lz - ghost - 2);
                f(12, i, j, lz - ghost) = f(bb(12), i + 2 * e(12, 0), j + 2 * e(12, 1), lz - ghost - 2);
                f(13, i, j, lz - ghost) = f(bb(13), i + 2 * e(13, 0), j + 2 * e(13, 1), lz - ghost - 2);
                f(16, i, j, lz - ghost) = f(bb(16), i + 2 * e(16, 0), j + 2 * e(16, 1), lz - ghost - 2);
                f(17, i, j, lz - ghost) = f(bb(17), i + 2 * e(17, 0), j + 2 * e(17, 1), lz - ghost - 2);
                f(20, i, j, lz - ghost) = f(bb(20), i + 2 * e(20, 0), j + 2 * e(20, 1), lz - ghost - 2);
                f(22, i, j, lz - ghost) = f(bb(22), i + 2 * e(22, 0), j + 2 * e(22, 1), lz - ghost - 2);
                f(23, i, j, lz - ghost) = f(bb(23), i + 2 * e(23, 0), j + 2 * e(23, 1), lz - ghost - 2);
                f(25, i, j, lz - ghost) = f(bb(25), i + 2 * e(25, 0), j + 2 * e(25, 1), lz - ghost - 2);
            });
    }
    // user boundary bounce back
    Kokkos::parallel_for(
        "usrbc", mdrange_policy4({0, ghost, ghost, ghost}, {q, lx - ghost, ly - ghost, lz - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            if (usr(i, j, k) == 0 && usr(i + e(ii, 0), j + e(ii, 1), k + e(ii, 2)) == 1)
            {
                f(ii, i, j, k) = f(bb(ii), i + 2 * e(ii, 0), j + 2 * e(ii, 1), k + 2 * e(ii, 2));
            }
        });
    // streaming process
    Kokkos::parallel_for(
        "stream1", mdrange_policy4({0, ghost, ghost, ghost}, {q, lx - ghost, ly - ghost, lz - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            ft(ii, i, j, k) = f(ii, i - e(ii, 0), j - e(ii, 1), k - e(ii, 2));
        });

    Kokkos::fence();

    Kokkos::parallel_for(
        "stream2", mdrange_policy4({0, ghost, ghost, ghost}, {q, lx - ghost, ly - ghost, lz - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            f(ii, i, j, k) = ft(ii, i, j, k);
        });
};

void LBM::Update()
{
    // update macroscopic value
    Kokkos::parallel_for(
        "initv", mdrange_policy3({0, 0, 0}, {lx, ly, lz}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            ua(i, j, k) = 0;
            va(i, j, k) = 0;
            wa(i, j, k) = 0;
            p(i, j, k) = 0;
        });
    Kokkos::fence();
    for (int k = ghost; k < lz - ghost; k++)
    {
        for (int j = ghost; j < ly - ghost; j++)
        {
            for (int i = ghost; i < lx - ghost; i++)
            {
                for (int ii = 0; ii < 27; ii++)
                {
                    p(i, j, k) = p(i, j, k) + f(ii, i, j, k) / 3.0;

                    ua(i, j, k) = ua(i, j, k) + f(ii, i, j, k) * e(ii, 0);
                    va(i, j, k) = va(i, j, k) + f(ii, i, j, k) * e(ii, 1);
                    wa(i, j, k) = wa(i, j, k) + f(ii, i, j, k) * e(ii, 2);
                }

                ua(i, j, k) = ua(i, j, k) * usr(i, j, k);
                va(i, j, k) = va(i, j, k) * usr(i, j, k);
                wa(i, j, k) = wa(i, j, k) * usr(i, j, k);
                p(i, j, k) = p(i, j, k) * usr(i, j, k);
                if (x_lo == 0)
                {
                    ua(ghost, j, k) = u0 * 4.0 * (z_lo + k - ghost) * (glz - 1 - (z_lo + k - ghost)) / pow((glz - 1), 2);
                    va(ghost, j, k) = 0.0;
                    wa(ghost, j, k) = 0.0;
                }
                if (x_hi == glx)
                {
                    p(lx - ghost - 1, j, k) = 0.0;
                }
            }
        }
    }
    Kokkos::fence();
};

void LBM::MPIoutput(int n)
{
    // MPI_IO
    MPI_File fh;
    MPIO_Request request;
    MPI_Status status;
    MPI_Offset offset = 0;

    MPI_Datatype FILETYPE, DATATYPE;
    // buffer
    int tp;
    float ttp;
    double fp;
    // min max
    double umin, umax, wmin, wmax, vmin, vmax, pmin, pmax;
    // transfer
    double *uu, *vv, *ww, *pp, *xx, *yy, *zz;
    int start[3];
    uu = (double *)malloc(ex * ey * ez * sizeof(double));
    vv = (double *)malloc(ex * ey * ez * sizeof(double));
    ww = (double *)malloc(ex * ey * ez * sizeof(double));
    pp = (double *)malloc(ex * ey * ez * sizeof(double));
    xx = (double *)malloc(ex * ey * ez * sizeof(double));
    yy = (double *)malloc(ex * ey * ez * sizeof(double));
    zz = (double *)malloc(ex * ey * ez * sizeof(double));

    for (int k = 0; k < ez; k++)
    {
        for (int j = 0; j < ey; j++)
        {
            for (int i = 0; i < ex; i++)
            {

                uu[i + j * ex + k * ey * ex] = ua(i + ghost, j + ghost, k + ghost);
                vv[i + j * ex + k * ey * ex] = va(i + ghost, j + ghost, k + ghost);
                ww[i + j * ex + k * ey * ex] = wa(i + ghost, j + ghost, k + ghost);
                pp[i + j * ex + k * ey * ex] = p(i + ghost, j + ghost, k + ghost);
                xx[i + j * ex + k * ey * ex] = (double)4.0 * (x_lo + i) / glx;
                yy[i + j * ex + k * ey * ex] = (double)(y_lo + j) / gly;
                zz[i + j * ex + k * ey * ex] = (double)(z_lo + k) / glz;
            }
        }
    }

    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {lx - ghost, ly - ghost, lz - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = ua(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(umax));

    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {lx - ghost, ly - ghost, lz - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = va(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(vmax));

    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {lx - ghost, ly - ghost, lz - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = wa(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(wmax));

    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {lx - ghost, ly - ghost, lz - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = p(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(pmax));

    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {lx - ghost, ly - ghost, lz - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = ua(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(umin));

    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {lx - ghost, ly - ghost, lz - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = va(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(vmin));

    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {lx - ghost, ly - ghost, lz - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = wa(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(wmin));

    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {lx - ghost, ly - ghost, lz - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = p(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(pmin));

    std::string str1 = "output" + std::to_string(n) + ".plt";
    const char *na = str1.c_str();
    std::string str2 = "#!TDV112";
    const char *version = str2.c_str();
    MPI_File_open(MPI_COMM_WORLD, na, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

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
        tp = 7;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 120;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 121;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 122;
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
        tp = 119;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 112;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 20+15*4=80
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

        // 80 + 10 * 4 = 120

        // Strand id
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SOLUTION TIME
        double nn = (double)n;
        fp = nn;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE COLOR
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE TYPE
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SPECIFY VAR LOCATION
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ARE RAW LOCAL
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // NUMBER OF MISCELLANEOUS
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ORDERED ZONE
        tp = glx;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = gly;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = glz;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // AUXILIARY
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 120 + 13 * 4 = 172
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
        // 172 + 12 * 4 = 220
        // MIN AND MAX VALUE FLOAT 64
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 4.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 1.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 1.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = umin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = umax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = wmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = wmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = pmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = pmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);

        // 220 + 14 * 8 = 332
    }

    offset = 332;

    int loclen[3] = {ex, ey, ez};
    int glolen[3] = {glx, gly, glz};
    int iniarr[3] = {0, 0, 0};
    int localstart[3] = {x_lo, y_lo, z_lo};

    MPI_Type_create_subarray(dim, glolen, loclen, localstart, MPI_ORDER_FORTRAN, MPI_DOUBLE, &DATATYPE);

    // MPI_Type_commit(&DATATYPE);

    MPI_Type_contiguous(7, DATATYPE, &FILETYPE);

    MPI_Type_commit(&FILETYPE);

    MPI_File_set_view(fh, offset, MPI_DOUBLE, FILETYPE, "native", MPI_INFO_NULL);

    MPI_File_write_all(fh, xx, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, yy, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, zz, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, uu, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, vv, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, ww, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, pp, ex * ey * ez, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);

    free(uu);
    free(vv);
    free(ww);
    free(pp);
    free(xx);
    free(yy);
    free(zz);

    MPI_Barrier(MPI_COMM_WORLD);
};

void LBM::Output(int n)
{
    std::ofstream outfile;
    std::string str = "output" + std::to_string(n) + std::to_string(comm.me);
    outfile << std::setiosflags(std::ios::fixed);
    outfile.open(str + ".dat", std::ios::out);

    outfile << "variables=x,y,z,f" << std::endl;
    outfile << "zone I=" << lx - 2 << ",J=" << ly - 2 << ",K=" << lz - 2 << std::endl;

    for (int k = 1; k < lz - 1; k++)
    {
        for (int j = 1; j < ly - 1; j++)
        {
            for (int i = 1; i < lx - 1; i++)
            {

                outfile << std::setprecision(8) << setiosflags(std::ios::left) << x_lo + i - 2 << " " << y_lo + j - 2 << " " << z_lo + k - 2 << " " << f(1, i, j, k) << std::endl;
            }
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
