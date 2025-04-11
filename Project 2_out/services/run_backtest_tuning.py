import numpy as np
import sys
sys.path.append("./")
print(sys.path)
from services.hyperparameter_tuning import *
# NOTE: this script should not be run, this was only for us to turne and test the model
# Grid searches to past in
trans_cost = 800
tw = 1
alph=0.95
ta= np.linspace(0,100,100)
rlambda= 100
rh = 0.5

hyperparam_tune(turnover_weight=tw, alpha=alph, tau=ta, reg_lambda=rlambda, 
    transaction_cost_weight=trans_cost, rho=rh)

# Calculating composite score
# sharpe = np.array([0.2289, 0.2002, 0.2053, 0.2099, 0.2234, 0.1729, 0.1737,0.1875, 0.1955, 0.2072,0.1852, 0.2074, 0.1885, 0.2070, 0.2074])
# turn = np.array([0.2723, 0.1980, 0.2080, 0.2168, 0.2049, 0.1646, 0.1370, 0.1514, 0.1214, 0.1379, 0.1483, 0.1354, 0.1804, 0.1441, 0.1329])
# result = sharpe - 0.25 * turn
# max_index, max_value = max(enumerate(result), key=lambda x: x[1])

# print(f"Max value: {max_value}, Index: {max_index}")
# print(result)

# print(sharpe[12], turn[12], result[12])