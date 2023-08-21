#include <math.h>
#include <cmath>
#include <pcg_random.hpp>
#include <random>
#include <vector>
#include "BS_thread_pool.hpp"
using namespace std;

namespace pimc
{
    class pimc_continuous
    {
    public:
        double beta = 0;           //// inverse T

        int N = 0;                 /// system size

        double h = 0;              // transverse field
        double co_j = 0;           // ising interaction coefficient
        vector<double> H;          ///
        vector<vector<double>> J;  /// pair_interaction
        BS::thread_pool pool;

        pimc_continuous(vector<vector<pair<double, int>>>& s, int n, double h, vector<double>& H,
                        vector<vector<double>>& J, double beta, int cnt_thread, pcg32& gen)
        {
            init(s, n, h, H, J, beta, cnt_thread, gen);
        }

        void init(vector<vector<pair<double, int>>>& s, int _n, double _h, vector<double>& _H,
                  vector<vector<double>>& _J, double _beta, int _cnt_thread, pcg32& gen)
        {
            N = _n;
            h = _h;
            beta = _beta;
            H = _H;
            J = _J;
            pool.reset(_cnt_thread);
            s.resize(N);
            uniform_real_distribution<> uni_random(0, 1);
            for (int i = 0; i < N; i++)
            {
                s[i].push_back({0, -2});
                int c = (uni_random(gen) < 0.5);
                s[i].push_back({beta, 2 * c - 1});
            }
        }
        void reset(vector<vector<pair<double, int>>>& s, pcg32& gen)
        {
            uniform_real_distribution<> uni_random(0, 1);
            s.clear();
            s.resize(N);
            for (int i = 0; i < N; i++)
            {
                s[i].push_back({0, -2});
                int c = (uni_random(gen) < 0.5);
                s[i].push_back({beta, 2 * c - 1});
            }
            h = 1;
            co_j = 0;
        }
        void annealing(double t)
        {
            h = 1 - t;
            co_j = t;
        }

        void generate_kinks(vector<vector<pair<double, int>>>& s, pcg32& gen)
        {
            exponential_distribution<> exp_dis(h);

            for (int i = 0; i < N; i++)
            {
                double t = 0;
                while (true)
                {
                    t += exp_dis(gen);
                    if (t > beta) break;
                    s[i].push_back({t, 0});
                }

                sort(s[i].begin(), s[i].end());

                for (int j = 0; j < s[i].size(); j++)
                {
                    if (s[i][j].second == 0) s[i][j].second = s[i][j + 1].second;
                }
            }
        }

        void remove_sudo_kinks(vector<vector<pair<double, int>>>& s, pcg32& gen)
        {
            vector<vector<pair<double, int>>> new_s(N);

            for (int i = 0; i < s.size(); i++)
            {
                for (int j = 0; j < s[i].size(); j++)
                {
                    int k = j;
                    while (k + 1 < s[i].size() && s[i][k + 1].second == s[i][k].second)
                    {
                        k++;
                    }

                    new_s[i].push_back(s[i][k]);
                    j = k;
                }
            }

            swap(s, new_s);
        }

        double get_inter(vector<vector<pair<double, int>>>& s, vector<int>& last, int x, int y)
        {
            double res = 0;

            double st_x = s[x][last[x]].first;
            double ed_x = s[x][last[x] + 1].first;

            int spin_x = s[x][last[x] + 1].second;
            while (last[y] < s[y].size())
            {
                if (s[y][last[y]].first < st_x)
                    last[y]++;
                else if (s[y][last[y]].first < ed_x)
                {
                    double dtau = s[y][last[y]].first - max(s[y][last[y] - 1].first, st_x);
                    res += dtau * s[y][last[y]].second * spin_x * J[x][y];
                    last[y]++;
                }
                else
                {
                    double dtau = ed_x - max(s[y][last[y] - 1].first, st_x);
                    res += dtau * s[y][last[y]].second * spin_x * J[x][y];
                    break;
                }
            }
            return res;
        }

        void single_site_update(vector<vector<pair<double, int>>>& s, int x, pcg32& gen)
        {
            uniform_real_distribution<> uni_random(0, 1);
            vector<int> last(N, 1);
            last[x] = 0;
            for (int i = 1; i < s[x].size(); i++)
            {
                double dE = 0;
                for (int j = 0; j < N; j++)
                {
                    if (x == j) continue;
                    dE += get_inter(s, last, x, j);
                }

                dE += H[x] * s[x][i].second * (s[x][i].first - s[x][i - 1].first);

                dE *= -2 * co_j;

                if (uni_random(gen) < exp(-dE))
                {
                    s[x][i].second *= -1;
                }

                last[x] = i;
            }
        }
        void single_flip_update(vector<vector<pair<double, int>>>& s, pcg32& gen)
        {
            for (int i = 0; i < N; i++)
            {
                single_site_update(s, i, gen);
            }
        }

        void cluster_update(vector<vector<pair<double, int>>>& s, pcg32& gen)
        {
            generate_kinks(s, gen);
            single_flip_update(s, gen);
            remove_sudo_kinks(s, gen);
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

        void get_energy_random(vector<vector<pair<double, int>>>& s, pcg32& gen, vector<int>& state, double& ene)
        {
            uniform_real_distribution<> uni_random(0, beta);
            int trial_step = 100;

            for (int i = 0; i < trial_step; i++)
            {
                pair<double, int> tau = {uni_random(gen), -2};
                vector<int> cur;
                for (int j = 0; j < N; j++)
                {
                    auto t = lower_bound(s[j].begin(), s[j].end(), tau);

                    cur.push_back(t->second);
                }
                double E = get_energy(cur);
                if (E < ene)
                {
                    ene = E;
                    swap(state, cur);
                }
            }
        }
    };

}  // namespace pimc