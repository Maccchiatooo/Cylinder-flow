#include "lbm.hpp"

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
            ua(i, j) = 0;
            va(i, j) = 0;
            p(i, j) = 0;
            rho(i, j) = rho0;

            usr(i, j) = 1;
            if (pow((i - 0.25 * (lx - 5)), 2) + pow((j - 0.5 * (ly - 1)), 2) <= 0.01 * (ly - 5) * (ly - 5))
            {
                usr(i, j) = 0;
            }

            ua(i, j) = (this->u0 * 4.0 * (j - 2) * (ly - 3 - j) / pow((ly - 5), 2)) * usr(i, j) * (1.0 + ran(i, j) * 0.001);
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
        "collision", mdrange_policy3({0, 2, 2}, {q, lx - 2, ly - 2}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            double feq = t(ii) * p(i, j) * 3.0 +
                         t(ii) * (3.0 * (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                  4.5 * pow(e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j), 2) -
                                  1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));
            f(ii, i, j) -= (f(ii, i, j) - feq) / (tau0 + 0.5);
        });
};
void LBM::Streaming()
{
    Kokkos::parallel_for(
        "stream3", range_policy(1, ly - 1), KOKKOS_CLASS_LAMBDA(const int j) {
            f(1, 1, j) = f(1, 2, j);
            f(5, 1, j) = f(5, 2, j);
            f(8, 1, j) = f(8, 2, j);

            f(3, lx - 2, j) = f(3, lx - 3, j);
            f(7, lx - 2, j) = f(7, lx - 3, j - 1);
            f(6, lx - 2, j) = f(6, lx - 3, j + 1);
        });

    Kokkos::parallel_for(
        "stream4", range_policy(2, lx - 2), KOKKOS_CLASS_LAMBDA(const int i) {
            f(2, i, 1) = f(4, i, 3);
            f(5, i, 1) = f(7, i + 2, 3);
            f(6, i, 1) = f(8, i - 2, 3);

            f(4, i, ly - 2) = f(2, i, ly - 4);
            f(7, i, ly - 2) = f(5, i - 2, ly - 4);
            f(8, i, ly - 2) = f(6, i + 2, ly - 4);
        });

    Kokkos::parallel_for(
        "usrbc", mdrange_policy3({0, 0, 0}, {q, lx, ly}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            if (usr(i, j) == 0 && usr(i + e(ii, 0), j + e(ii, 1)) == 1)
            {
                f(ii, i, j) = f(bb(ii), i + 2 * e(ii, 0), j + 2 * e(ii, 1));
            }
        });

    Kokkos::parallel_for(
        "stream1", mdrange_policy3({0, 1, 1}, {q, lx - 1, ly - 1}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            ft(ii, i, j) = f(ii, i - e(ii, 0), j - e(ii, 1));
        });

    Kokkos::parallel_for(
        "stream2", mdrange_policy3({0, 2, 2}, {q, lx - 2, ly - 2}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            f(ii, i, j) = ft(ii, i, j);
        });
};

void LBM::Update()
{
    Kokkos::parallel_for(
        "initv", mdrange_policy2({0, 0}, {lx, ly}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            ua(i, j) = 0;
            va(i, j) = 0;
            p(i, j) = 0;
        });

    for (int j = 2; j < this->ly - 2; j++)
    {
        for (int i = 2; i < this->lx - 2; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {
                p(i, j) = p(i, j) + f(ii, i, j) / 3.0;

                ua(i, j) = ua(i, j) + f(ii, i, j) * e(ii, 0);
                va(i, j) = va(i, j) + f(ii, i, j) * e(ii, 1);
            }

            ua(i, j) = ua(i, j) * usr(i, j);
            va(i, j) = va(i, j) * usr(i, j);
            p(i, j) = p(i, j) * usr(i, j);

            ua(2, j) = this->u0 * 4.0 * (j - 2) * (ly - 3 - j) / pow((ly - 5), 2);
            va(2, j) = 0.0;
            p(lx - 3, j) = 0.0;
        }
    }
};
void LBM::Output(int n)
{
    std::ofstream outfile;
    std::string str = "output" + std::to_string(n);
    outfile << std::setiosflags(std::ios::fixed);
    outfile.open(str + ".dat", std::ios::out);

    outfile << "variables=x,y,u,v,p" << std::endl;
    outfile << "zone I=" << this->lx - 4 << ",J=" << this->ly - 4 << std::endl;

    for (int j = 2; j < this->ly - 2; j++)
    {
        for (int i = 2; i < this->lx - 2; i++)
        {

            outfile << std::setprecision(8) << setiosflags(std::ios::left) << 4 * (i - 2.0) / (lx - 5.0) << " " << (j - 2.0) / (ly - 5.0) << " " << ua(i, j) << " " << va(i, j) << " " << p(i, j) << std::endl;
        }
    }

    outfile.close();
    printf("\n");
    printf("The result %d is writen\n", n);
    printf("\n");
    printf("============================\n");
};
