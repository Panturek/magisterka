from desaggregation import Desaggregation as DA
import pandas as pd
import numpy as np

cri = pd.DataFrame([[1, 2], [2, 3], [3, 2]])
dem = pd.DataFrame([1, 2, 3])
names = ['y', 'x1', 'x2']

model = DA(cri, dem, names)

model.alpha_i = 5
model.alpha = 5
model.cost_criteria = [1,]

model.set_levels([[0.1, 0.5], [0.0, 0.8]], [0.2, 0.4])

expected_cri_levels = [ [0.1, 0.8],
                        [0.2, 0.6],
                        [0.3, 0.4],
                        [0.4, 0.2],
                        [0.5, 0.0]]
expected_dem_levels = [ 0.2, 0.25, 0.3, 0.35, 0.4 ]
expected_c_ijk_coeffs = {(0, 0, 0): 1, (1, 0, 0): 0, (0, 0, 1): 1,
                         (1, 0, 1): 0, (0, 0, 2): 1, (1, 0, 2): 1,
                         (0, 0, 3): 1, (1, 0, 3): 1, (0, 1, 0): 1,
                         (1, 1, 0): 0, (0, 1, 1): 1, (1, 1, 1): 0,
                         (0, 1, 2): 1, (1, 1, 2): 0, (0, 1, 3): 1,
                         (1, 1, 3): 0}
expected_c_jm_coeffs = {(0, 0): 1, (0, 1): 1, (0, 2): 1,
                        (0, 3): 1, (1, 0): 1, (1, 1): 1,
                        (1, 2): 1, (1, 3): 1}



assert( np.allclose( model.cri_lvls,
        expected_cri_levels ))
assert( np.allclose( model.dem_lvls,
        expected_dem_levels ))

assert( np.allclose( model.interpolate(0, 0.5, 1), 0.5) )
assert( np.allclose( model.interpolate(0, 2, 1), 1.) )
assert( np.allclose( model.interpolate(0, -2, 1), 0) )

assert( model.get_criteria_coeffs().keys() == expected_c_ijk_coeffs.keys())
assert( model.get_demand_coeffs().keys() == expected_c_jm_coeffs.keys())

assert (np.allclose(list(model.get_criteria_coeffs().values()),
                    list(expected_c_ijk_coeffs.values() )))
assert( np.allclose(list( model.get_demand_coeffs().values()),
                    list( expected_c_jm_coeffs.values() )))