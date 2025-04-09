from services.main import *

trans_cost = [0, 100, 100]
tw = 1
alph=0.95
ta=1000
rlambda=0.01

hyperparam_tune(turnover_weight=tw, alpha=alph, tau=ta, reg_lambda=rlambda, 
    transaction_cost_weight=trans_cost)