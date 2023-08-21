#include <cmath>
#include <pcg_random.hpp>
#include <random>
#include <vector>
#include "BS_thread_pool.hpp"
using namespace std;

namespace pimc
{
    class pimc_model
    {
    public:
        double beta = 0;  //// inverse T
        int p = 0;        /// slice

        double beta_over_p = 0.1;
        double tau = 0;            /// trotter slice
        int N = 0;                 /// system size
        double c_p = 0;            // model parameter
        double k_p = 0;
        double h = 0;              // transverse field
        double co_j = 0;           // ising interaction coefficient
        vector<double> H;          ///
        vector<vector<double>> J;  /// pair_interaction
        BS::thread_pool pool;

        pimc_model(vector<vector<int>>& s, int n, double h, vector<double>& H, vector<vector<double>>& J, double beta,
                   int cnt_thread, vector<pcg32>& gen)
        {
            init(s, n, h, H, J, beta, cnt_thread, gen);
        }

        void init(vector<vector<int>>& s, int _n, double _h, vector<double>& _H, vector<vector<double>>& _J,
                  double _beta, int _cnt_thread, vector<pcg32>& gen)
        {
            N = _n;
            h = _h;
            beta = _beta;
            H = _H;
            J = _J;
            pool.reset(_cnt_thread);
            p = (int)(beta / beta_over_p);
            tau = beta / p;
            gen.resize(p);
            for (int i = 0; i < p; i++)
            {
                gen[i] = pcg32(123456, i);
            }
            s.resize(p, vector<int>(N));
            uniform_real_distribution<> uni_random(0, 1);

            for (int i = 0; i < p; i++)
            {
                double t = uni_random(gen[0]);
                for (int j = 0; j < N; j++)
                {
                    if (t > 0.5)
                    {
                        s[i][j] = 1;
                    }
                    else
                        s[i][j] = -1;
                }
            }

            // c_p = pow(0.5 * sinh(2 * h * beta / p), p * N / 2.0);
            k_p = 0.5 * log(1.0 / tanh(h * beta / p));
        }

        void reset(vector<vector<int>>& s, pcg32& gen)
        {
            uniform_real_distribution<> uni_random(0, 1);

            for (int i = 0; i < p; i++)
            {
                double t = uni_random(gen);
                for (int j = 0; j < N; j++)
                {
                    if (t > 0.5)
                    {
                        s[i][j] = 1;
                    }
                    else
                        s[i][j] = -1;
                }
            }
            h = 1;
            co_j = 0;
            k_p = 0.5 * log(1.0 / tanh(h * beta / p));
        }

        void annealing(double t)
        {
            h = 1 - t;
            co_j = t;
            // c_p = pow(0.5 * sinh(2 * h * beta / p), p * N / 2.0);
            k_p = 0.5 * log(1.0 / tanh(h * beta / p));
        }
        double get_ising_energy(vector<int>& s, int x)
        {
            double res = 0;
            for (int i = 0; i < N; i++)
            {
                // if (i == x) continue;
                res += J[x][i] * s[i];
            }
            res += H[x];

            res *= s[x] * co_j;

            return res;
        }

        void single_slice_update(vector<vector<int>>& s, int i, pcg32& gen)
        {
            uniform_real_distribution<> uni_random(0, 1);
            for (int j = 0; j < N; j++)
            {
                double dE = get_ising_energy(s[i], j) * tau;
                if (i == 0)
                {
                    dE += k_p * s[i][j] * s[i + 1][j];
                }
                else if (i == p - 1)
                {
                    dE += k_p * s[i][j] * s[i - 1][j];
                }
                else
                {
                    dE += k_p * s[i][j] * (s[i - 1][j] + s[i + 1][j]);
                }
                dE *= -2;

                if (uni_random(gen) < exp(-dE))
                {
                    s[i][j] *= -1;
                }
            }
        }
        void single_flip_update(vector<vector<int>>& s, vector<pcg32>& gen)
        {
            for (int i = 0; i < p; i += 2)
            {
                pool.push_task(&pimc_model::single_slice_update, this, ref(s), i, ref(gen[i]));
            }

            pool.wait_for_tasks();

            for (int i = 1; i < p; i += 2)
            {
                pool.push_task(&pimc_model::single_slice_update, this, ref(s), i, ref(gen[i]));
            }
            pool.wait_for_tasks();
        }

        /////// for measurements

        double get_energy(vector<int>& s)
        {
            double res = 0;
            for (int i = 0; i < N; i++)
            {
                double cur = 0;
                for (int j = 0; j < i; j++)
                {
                    cur += J[i][j] * s[j];
                    // res += J[i][j] * s[i] * s[j];
                }
                cur *= s[i];
                res += cur;

                res += H[i] * s[i];
            }
            return res;
        }
    };

}  // namespace pimc