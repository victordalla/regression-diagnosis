"""
    Constants used across all files
"""

# The constants in this file must be defined and checked by the user of the template

import numpy as np

from typing import Dict, List, Optional
from copy import deepcopy
from datetime import date


def from_dict_to_list(dictionary):
    from itertools import chain
    return list(chain(*dictionary.values()))

seed = 0
true_params = {"b0": 6, "b1": 1.3, "scale": 50}
n = 300
x_min = -50
x_max = 100

np.random.seed(seed)
x_fix = np.random.uniform(x_min, x_max, n)
np.random.seed(seed)
e_fix = np.random.normal(0, true_params["scale"], n)
np.random.seed(seed)
y_fix = true_params["b0"] + true_params["b1"] * x_fix + e_fix