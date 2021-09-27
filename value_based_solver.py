from pymprog import *
from desaggregation import *

class ValueBasedSolver:
    def __init__(self, data):
        self._m = data
        self.post_opt = {}
        self.post_opt_w = {}

    def infer_b_values(self, _w_):
        b = []
        for i in _w_.values():
            b.append( sum([ i[t].primal + self._m.gamma_i
                        for t in range( len(i) ) ])
                    )
        return np.array(b)

    def infer_x_values(self, _w_, _b_):
        x = {}
        for i in range(len(_w_)):
            for k in range(len(_w_[i]) + 1):
                x[i, k] = sum([_w_[i][t].primal + self._m.gamma_i for t in range(k)]) / _b_[i]
        return x

    def infer_y_values(self, _z_, ):
        y = {}
        y[0] = 0
        for m in range(0, len(_z_)):
            y[m+1] = sum([_z_[t].primal + self._m.gamma for t in range(m+1)])
        return y

    def infer_all_values(self, err_m, err_p, w, z):
        T = self._m.time_horizon

        self.b = self.infer_b_values(w)
        self.x_vals = self.infer_x_values(w, self.b)
        self.y_vals = self.infer_y_values(z)
        self.sigm = [ -err_m[i].primal for i in range(T) ]
        self.sigp = [ err_p[i].primal for i in range(T) ]

    def set_constraints(self, err_m, err_p, w, z):
        T = self._m.time_horizon
        n = self._m.num_cri
        alpha = self._m.alpha
        alphai = self._m.alpha_i
        gamma = self._m.gamma
        gammai = self._m.gamma_i
        c_ijk = self._m.get_criteria_coeffs()
        c_jm = self._m.get_demand_coeffs()

        for j in range( T ):
            w_sum = sum([
                c_ijk[(i, j, k)] * w[i][k]
                for i in range(n) for k in range( alphai - 1 )
                ])

            z_sum = sum([
                c_jm[(j, m)] * z[ m ] for m in range( alpha - 1 )
                ])

            gik_sum = sum([
                gammai * sum([ c_ijk[(i, j, k)] for k in range( alphai - 1 ) ])
                for i in range(n)
            ])

            gm_sum = sum([
                gamma * c_jm[(j, m)] for m in range(alpha - 1)
            ])

            LHS = w_sum - z_sum
            RHS = gm_sum - gik_sum
            LHS - err_p[j] + err_m[j] == RHS

        # formula 18, second constraint(s)
        sum(z) == 1 - gamma * (alpha - 1)

        # formula 18, third constraint(s)
        LHS = sum( [ sum(w[i]) for i in range(n) ] )
        RHS = 1 - sum( [ gammai * (alphai - 1) for i in range(n)] )
        LHS == RHS

    def solve(self, verboseOutput=False, postOptimize=None):
        begin('value-based model')
        verbose(verboseOutput)
        solver('simplex')

        T = self._m.time_horizon
        alpha = self._m.alpha
        alphai = self._m.alpha_i

        err_p = var('err_p', T)
        err_m = var('err_m', T)
        w = {}
        for i in range(self._m.num_cri):
            w[i] = var('w' + str(i), alphai - 1)
        z = var('z', alpha - 1)
        F = sum(err_m) + sum(err_p)

        minimize(F)
        self.set_constraints(err_m, err_p, w, z)

        solve()
        self.infer_all_values(err_m, err_p, w, z)

        if postOptimize is not None:
            operation, i, K = postOptimize
            opt_F = sum([err_m[j].primal + err_p[j].primal for j in range(T)])
            g = self._m.gamma_i
            F_prim = sum([w[i][k] + g for k in range(K)])

            if operation is "min":
                minimize(F_prim)
            if postOptimize is "max":
                maximize(F_prim)
            
            r_ = F_prim <= (1 + self._m.epsilon) * opt_F
            solve()
            r_.delete()

            __b = self.infer_b_values(w)
            self.post_opt[i, K, 'b', operation] = __b
            self.post_opt[i, K, 'x', operation] = self.infer_x_values(w, __b)
            if i not in self.post_opt_w.keys():
                self.post_opt_w[i] = []
            self.post_opt_w[i].append([w[i][k].primal for k in range(K)])
        end()

    def get_post_opt_solutions(self):
        bs, xs = [], []
        for key in self.post_opt.keys():
            if 'b' in key:
                bs.append(np.array(self.post_opt[key]))
            if 'x' in key:
                xs.append(self.post_opt[key])
        return bs, xs

    def get_postopt_b(self):
        bs, xs = self.get_post_opt_solutions()
        n_bs = np.array(bs)
        b_min = n_bs.min(axis=0)
        b_mean = n_bs.mean(axis=0)
        b_max = n_bs.max(axis=0)

        return b_min, b_mean, b_max

    def get_postopt_x(self):
        bs, xs = self.get_post_opt_solutions()
        x_min, x_mean, x_max = {}, {}, {}
        aggr_xs = {}

        for i in range(self._m.num_cri):
            for k in range(self._m.alpha_i):
                aggr_xs[i, k] = np.array([x[i, k] for x in xs])

        for i in range(self._m.num_cri):
            for k in range(self._m.alpha_i):
                x_min[i, k] = aggr_xs[i, k].min()
                x_mean[i, k] = aggr_xs[i, k].mean()
                x_max[i, k] = aggr_xs[i, k].max()
        return x_min, x_mean, x_max

    def get_postopt_y(self, x_min, x_mean, x_max):
        Y = {}
        y_min, y_mean, y_max = [], [], []
        def yappend(i, Y, X, b, ai):
            Y.append([X[i, k] * b[i] for k in range(ai)])

        for i in range(self._m.num_cri):
            yappend(i, y_min, x_min, self.b, self._m.alpha_i)
            yappend(i, y_mean, x_mean, self.b, self._m.alpha_i)
            yappend(i, y_max, x_max, self.b, self._m.alpha_i)

        Y_min = np.array(y_min).sum(axis=0)
        Y_mean = np.array(y_mean).sum(axis=0)
        Y_max = np.array(y_max).sum(axis=0)

        return Y_min, Y_mean, Y_max
