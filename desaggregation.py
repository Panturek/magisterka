import numpy as np
import pandas as pd

class Desaggregation:
    "contains methods for desaggregation data processing"
    def __init__(self, criteria, demand, names):
        self.criteria = criteria
        self.demand = demand
        self.criteria_pct = criteria.pct_change().dropna()
        self.demand_pct = demand.pct_change().dropna()
        self.time_horizon = len(demand.index) - 1
        self.num_cri = len(criteria.columns)
        self.names = names
        self._params = {}

    def __getitem__(self, index):
        if index in self._params.keys():
            return self._params[index]
        return None

    def __setitem__(self, index, value):
        self._params[index] = value


    def _set_lvls(self, _range, count):
        return np.linspace(
            np.min(_range, axis=0),
            np.max(_range, axis=0),
            count)

    def _set_cost_criteria(self, cri_idcs):
        for idx in cri_idcs:
            self.cri_lvls.T[idx] = np.flip(self.cri_lvls.T[idx])

    def set_levels(self, criteria_ranges, demand_ranges):
        self.cri_lvls = self._set_lvls(np.array(criteria_ranges).T, self.alpha_i )
        self.dem_lvls = self._set_lvls(np.array(demand_ranges), self.alpha )
        if self.cost_criteria:
            self._set_cost_criteria(self.cost_criteria)

    def interpolate(self, v_k, V, v_kx1):
        if V <= v_k:
            return 0
        if V >= v_kx1:
            return 1
        return (V - v_k) / (v_kx1 - v_k)

    def get_criteria_coeffs(self):
        criteria = self.criteria_pct.to_numpy()

        __coeffs = {}
        for j in range( self.time_horizon ):
            for k in range( self.alpha_i - 1 ):
                for i in range(self.num_cri ):
                    xi_k = self.cri_lvls[k, i]
                    xi_kx1 = self.cri_lvls[k + 1, i]
                    Xij = criteria[j, i]
                    __coeffs[(i, j, k)] = self.interpolate(xi_k, Xij, xi_kx1)
        return __coeffs

    def get_demand_coeffs(self):
        demand = self.demand_pct.to_numpy()

        __coeffs = {}
        for j in range(self.time_horizon ):
            for m in range(self.alpha - 1):
                y_m = self.dem_lvls[ m ]
                y_mx1 = self.dem_lvls[ m + 1 ]
                Yj = demand[j]
                __coeffs[(j, m)] = self.interpolate(y_m, Yj, y_mx1)
        return __coeffs
