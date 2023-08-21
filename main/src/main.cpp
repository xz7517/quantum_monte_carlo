#include <brute_solve.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <pcg_random.hpp>
#include <pimc_continuous.hpp>
#include <pimc_model.hpp>
#include <random>
#include <set>
#include <vector>

using namespace std;

double err = 1e-10;

void continuous_time_simulation(double t)
{
    double temperature = t;  // NOLINT
    int N = 0;               // atoi(argv[1]);  //system size
    double h = 1.0;          // atof(argv[2]);  //transverse field
    vector<double> H;
    vector<vector<double>> J;
    vector<vector<pair<double, int>>> s;  ///  first dimension is number of slices, second is size of system

    vector<int> annealing_steps = {100, 200, 400, 800, 1600, 3200, 6400, 12800};

    int mc_steps = 1;
    pcg_extras::seed_seq_from<std::random_device> seed_source;

    int seed = 1234567;
    pcg32 gen_init(seed_source);
    vector<pcg32> gen;

    ifstream infile("coupling_matrix.txt");

    if (!infile.is_open())
    {
        cout << "err opening the file" << endl;
        return;
    }

    infile >> N;
    cout << "N = " << N << endl;

    J.resize(N, vector<double>(N));
    H.resize(N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            infile >> J[i][j];
            J[j][i] = J[i][j];
        }
    }
    for (int i = 0; i < N; i++)
    {
        infile >> H[i];
    }
    infile.close();

    int thread = 1;
    auto m = pimc::pimc_continuous(s, N, h, H, J, 1.0 / temperature, thread, gen_init);

    ///////measurements
    set<vector<int>> ground_s;
    vector<vector<double>> ground_E(annealing_steps.size());
    double min_ene = 1e12;

    vector<vector<int>> true_ground_s;
    double true_min_ene = 1e12;

    /////// time measurement
    long long time_update = 0;
    long long time_measurement = 0;

    int iter_cnt = 10;
    double st_point = 0.3;
    for (auto l = 0; l < annealing_steps.size(); l++)
    {
        int annealing_step = annealing_steps[l];
        cout << "current step = " << annealing_step << endl;

        for (int k = 0; k < iter_cnt; k++)
        {
            min_ene = 1e12;
            m.reset(s, gen_init);
            for (auto i = 0; i < annealing_step; ++i)
            {
                auto t = st_point + static_cast<double>(i) / annealing_step * (1 - st_point);

                m.annealing(t);
                for (int i = 0; i < mc_steps; i++)
                {
                    ////update configuration
                    auto st_flip = chrono::steady_clock::now();
                    m.cluster_update(s, gen_init);
                    auto ed_flip = chrono::steady_clock::now();
                    time_update += chrono::duration_cast<chrono::nanoseconds>(ed_flip - st_flip).count();

                    ////do some measurements

                    if (t > 0.8)
                    {
                        auto st_measurement = chrono::steady_clock::now();
                        vector<int> state;
                        double ene = 1e9;
                        m.get_energy_random(s, gen_init, state, ene);

                        if (abs(ene - min_ene) < err)
                        {
                            ground_s.insert(state);
                        }
                        else if (ene < min_ene)
                        {
                            min_ene = ene;
                            ground_s.clear();
                            ground_s.insert(state);
                        }

                        auto ed_measurement = chrono::steady_clock::now();
                        time_measurement +=
                            chrono::duration_cast<chrono::nanoseconds>(ed_measurement - st_measurement).count();
                    }
                }
            }
            ground_E[l].push_back(min_ene);
        }
    }
    cout << "time_update = " << (double)time_update / 1e6 << endl;
    cout << "time_measurement = " << (double)time_measurement / 1e6 << endl;
    // brute_force_solve(m, true_min_ene, true_ground_s);
    //  classical_annealing(m, class_min_ene, class_ground_s);

    string str_beta = to_string((int)m.beta);
    ofstream data("continuous_beta=" + str_beta + ".txt");

    vector<double> std_E;
    vector<double> ave_E;
    for (auto x : ground_E)
    {
        double ave = 0;
        double var = 0;
        for (auto y : x)
        {
            data << y << " ";
            ave += y / iter_cnt;
        }
        data << endl;
        for (auto y : x)
        {
            var += (y - ave) * (y - ave) / iter_cnt;
        }
        std_E.push_back(sqrt(var));
        ave_E.push_back(ave);
    }
    data << endl;
    data << endl;
    for (auto x : ave_E) data << x << ", ";
    data << endl;
    for (auto x : std_E) data << x << ", ";
    data << endl;

    data << (double)time_update / 1e6 << endl;
    data.close();
}

void discrete_time_simulation(double t)
{
    double temperature = t;  // NOLINT
    int N = 0;               // atoi(argv[1]);  //system size
    double h = 1.0;          // atof(argv[2]);  //transverse field
    vector<double> H;
    vector<vector<double>> J;
    vector<vector<int>> s;  ///  first dimension is number of slices, second is size of system

    vector<int> annealing_steps = {100, 200, 400, 800, 1600, 3200, 6400, 12800};

    // for (auto& x : annealing_steps) x *= 3;
    int mc_steps = 1;
    pcg_extras::seed_seq_from<std::random_device> seed_source;

    int seed = 1234567;
    pcg32 gen_init(seed_source);
    vector<pcg32> gen;

    ifstream infile("coupling_matrix.txt");

    if (!infile.is_open())
    {
        cout << "err opening the file" << endl;
        return;
    }

    infile >> N;
    cout << "N = " << N << endl;

    J.resize(N, vector<double>(N));
    H.resize(N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            infile >> J[i][j];
            J[j][i] = J[i][j];
        }
    }
    for (int i = 0; i < N; i++)
    {
        infile >> H[i];
    }
    infile.close();

    int thread = 1;
    auto m = pimc::pimc_model(s, N, h, H, J, 1.0 / temperature, thread, gen);

    ///////measurements
    set<vector<int>> ground_s;
    vector<vector<double>> ground_E(annealing_steps.size());
    double min_ene = 1e9;

    /////// time measurement
    long long time_update = 0;
    long long time_measurement = 0;

    int iter_cnt = 10;
    for (auto l = 0; l < annealing_steps.size(); l++)
    {
        int annealing_step = annealing_steps[l];
        cout << "current step = " << annealing_step << endl;
        for (int k = 0; k < iter_cnt; k++)
        {
            m.reset(s, gen_init);
            min_ene = 1e9;
            for (auto i = 0; i < annealing_step; ++i)
            {
                auto t = static_cast<double>(i) / annealing_step;

                m.annealing(t);
                for (int i = 0; i < mc_steps; i++)
                {
                    ////update configuration
                    auto st_flip = chrono::steady_clock::now();
                    m.single_flip_update(s, gen);
                    auto ed_flip = chrono::steady_clock::now();
                    time_update += chrono::duration_cast<chrono::nanoseconds>(ed_flip - st_flip).count();

                    ////do some measurements

                    if (t > 0.8)
                    {
                        for (int k = 0; k < m.p; k++)
                        {
                            auto st_measurement = chrono::steady_clock::now();
                            vector<int> state;
                            double ene = m.get_energy(s[k]);

                            if (abs(ene - min_ene) < err)
                            {
                                ground_s.insert(s[k]);
                            }
                            else if (ene < min_ene)
                            {
                                min_ene = ene;
                                ground_s.clear();
                                ground_s.insert(s[k]);
                            }

                            auto ed_measurement = chrono::steady_clock::now();
                            time_measurement +=
                                chrono::duration_cast<chrono::nanoseconds>(ed_measurement - st_measurement).count();
                        }
                    }
                }
            }
            ground_E[l].push_back(min_ene);
        }
        // ground_E.push_back(cur / iter_cnt);
    }
    cout << "time_update = " << (double)time_update / 1e6 << endl;
    cout << "time_measurement = " << (double)time_measurement / 1e6 << endl;
    // brute_force_solve(m, true_min_ene, true_ground_s);
    //  classical_annealing(m, class_min_ene, class_ground_s);

    string str_beta = to_string((int)m.beta);
    ofstream data("discrete_beta=" + str_beta + ".txt");

    cout << "quantum annealing ene = " << endl;
    vector<double> std_E;
    vector<double> ave_E;
    for (auto x : ground_E)
    {
        double ave = 0;
        double var = 0;
        for (auto y : x)
        {
            data << y << " ";
            ave += y / iter_cnt;
        }
        data << endl;
        for (auto y : x)
        {
            var += (y - ave) * (y - ave) / iter_cnt;
        }
        std_E.push_back(sqrt(var));
        ave_E.push_back(ave);
    }
    data << endl;
    data << endl;
    for (auto x : ave_E) data << x << ", ";
    data << endl;
    for (auto x : std_E) data << x << ", ";
    data << endl;

    data << (double)time_update / 1e6 << endl;
    data.close();
}
int main()
{
    /////initializations
    vector<double> temperature = {0.05, 0.1, 0.2};
    for (double t : temperature) discrete_time_simulation(t);
    for (double t : temperature) continuous_time_simulation(t);
    cout << endl;

    return 0;
}