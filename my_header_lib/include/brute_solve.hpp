#include <cmath>
#include <vector>

using namespace std;

template <class MODEL>
void brute_force_solve(MODEL& m, double& min_ene, vector<vector<int>>& s)
{
    double err = 1e-10;
    int n = m.N;
    min_ene = 1e12;
    for (int i = 0; i < 1 << n; i++)
    {
        vector<int> cur(n);
        for (int j = 0; j < n; j++)
        {
            if ((i >> j) & 1)
                cur[j] = 1;
            else
                cur[j] = -1;
        }
        double ene = m.get_energy(cur);
        if (abs(ene - min_ene) < err)
        {
            s.push_back(cur);
        }
        else if (ene < min_ene)
        {
            min_ene = ene;
            s.clear();
            s.push_back(cur);
        }
    }

    return;
}