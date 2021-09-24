import numpy as np
import pandas as pd

class Desaggregation:
    "contains methods for desaggregation data processing"
    def __init__(self, criteria, demand, names):
        self.criteria = criteria
        self.demand = demand
        self.criteria_pct = criteria.pct_change()
        self.demand_pct = demand.pct_change()
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
            self.cri_lvls[idx].sort_series(ascending=False,
                                            inplace=True)


    def set_levels(self, criteria_ranges, demand_ranges):
        self.cri_lvls = self._set_lvls(criteria_ranges, self.alpha_i )
        self.dem_lvls = self._set_lvls(demand_ranges, self.alpha )
        if self.cost_criteria:
            self._set_cost_criteria(self.cost_criteria)
