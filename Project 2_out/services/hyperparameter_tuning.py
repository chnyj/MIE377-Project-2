# %% [markdown]
# 
# # MMF1921/MIE377 - Backtesting Template
# 
# The purpose of this program is to provide a template with which to develop Project 2. The project requires you to test different models  (and/or different model combinations) to create an asset management algorithm.
# 
# This template will be used by the instructor and TA to assess your trading algorithm using different datasets.
# 
# # PLEASE DO NOT MODIFY THIS TEMPLATE (for Project submission purposes)
#
# # 1. Read input files

# %%
import time
import math
from scipy.stats import gmean
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("./")
from services.project_function import *
import pandas as pd

def back_test_main(turnover_weight, alpha, tau,
                       reg_lambda, transaction_cost_weight, rho):
    adjClose = pd.read_csv("MIE377_AssetPrices_3.csv", index_col=0)
    factorRet = pd.read_csv("MIE377_FactorReturns_3.csv", index_col=0)

    # %%
    adjClose.index = pd.to_datetime(adjClose.index)
    factorRet.index = pd.to_datetime(factorRet.index)

    # %%
    # Initial budget to invest ($100,000)
    initialVal = 100000

    # Length of investment period (in months)
    investPeriod = 6

    # divide the factor returns by  100
    factorRet = factorRet/100

    #rf and factor returns
    riskFree = factorRet['RF']
    factorRet = factorRet.loc[:,factorRet.columns != 'RF'];


    # %%
    #Identify the tickers and the dates
    tickers = adjClose.columns
    dates   = factorRet.index

    # %%
    # Calculate the stocks monthly excess returns
    # pct change and drop the first null observation
    returns = adjClose.pct_change(1).iloc[1:, :]
    returns = returns  - np.diag(riskFree.values) @ np.ones_like(returns.values)
    # Align the price table to the asset and factor returns tables by discarding the first observation.
    adjClose = adjClose.iloc[1:,:]

    # %%
    assert adjClose.index[0] == returns.index[0]
    assert adjClose.index[0] == factorRet.index[0]

    # %% [markdown]
    # # 2. Run your program
    # 
    # This section will run your Project1_Function in a loop. The data will be loaded progressively as a growing window of historical observations.
    # Rebalancing will take place after every loop

    # %%
    # Start of out-of-sample test period
    testStart = returns.index[0] + pd.offsets.DateOffset(years=5)

    #End of the first investment period
    testEnd = testStart + pd.offsets.DateOffset(months=investPeriod) -  pd.offsets.DateOffset(days = 1)

    # End of calibration period
    calEnd = testStart -  pd.offsets.DateOffset(days = 1)

    # Total number of investment periods
    NoPeriods = math.ceil((returns.index[-1].to_period('M') - testStart.to_period('M')).n / investPeriod)

    # Number of assets
    n  = len(tickers)

    # Preallocate space for the portfolio weights (x0 will be used to calculate
    # the turnover rate)
    x  = np.zeros([n, NoPeriods])
    x0 = np.zeros([n, NoPeriods])

    # Preallocate space for the portfolio per period value and turnover
    currentVal = np.zeros([NoPeriods, 1])
    turnover   = np.zeros([NoPeriods, 1])

    #Initiate counter for the number of observations per investment period
    toDay = 0

    # Measure runtime: start the clock
    start_time = time.time()

    # Empty list to measure the value of the portfolio over the period
    portfValue = []

    for t in range(NoPeriods):
        # Subset the returns and factor returns corresponding to the current calibration period.
        periodReturns = returns[returns.index <= calEnd]
        periodFactRet = factorRet[factorRet.index <= calEnd]

        current_price_idx = (calEnd - pd.offsets.DateOffset(months=1) <= adjClose.index)&(adjClose.index <= calEnd)
        currentPrices = adjClose[current_price_idx]

        # Subset the prices corresponding to the current out-of-sample test period.
        periodPrices_idx = (testStart <= adjClose.index)&(adjClose.index <= testEnd)
        periodPrices = adjClose[periodPrices_idx]

        assert len(periodPrices) == investPeriod
        assert len(currentPrices) == 1
        # Set the initial value of the portfolio or update the portfolio value
        if t == 0:
            currentVal[0] = initialVal
        else:
            currentVal[t] = currentPrices @  NoShares.values.T
            #Store the current asset weights (before optimization takes place)
            x0[:,t] = currentPrices.values*NoShares.values/currentVal[t]

        #----------------------------------------------------------------------
        # Portfolio optimization
        # You must write code your own algorithmic trading function
        # The project function is in the services folder
        # Take in the period returns and period factor returns and produce
        # an allocation
        #----------------------------------------------------------------------
        x[:,t] = project_function(periodReturns, periodFactRet, tau)

        #Calculate the turnover rate
        if t > 0:
            turnover[t] = np.sum( np.abs( x[:,t] - x0[:,t] ) )

        # Number of shares your portfolio holds per stock
        NoShares = x[:,t]*currentVal[t]/currentPrices

        # Update counter for the number of observations per investment period
        fromDay = toDay
        toDay   = toDay + len(periodPrices)

        # Weekly portfolio value during the out-of-sample window
        portfValue.append(periodPrices@ NoShares.values.T)

        # Update your calibration and out-of-sample test periods
        testStart = testStart + pd.offsets.DateOffset(months=investPeriod)
        testEnd   = testStart + pd.offsets.DateOffset(months=investPeriod) - pd.offsets.DateOffset(days=1)
        calEnd    = testStart - pd.offsets.DateOffset(days=1)

    portfValue = pd.concat(portfValue, axis = 0)
    end_time = time.time()

    # %% [markdown]
    # # 3. Results

    # %%
    #--------------------------------------------------------------------------
    # 3.1 Calculate the portfolio average return, standard deviation, Sharpe ratio and average turnover.
    #-----------------------------------------------------------------------
    # Calculate the observed portfolio returns
    portfRets = portfValue.pct_change(1).iloc[1:,:]

    # Calculate the portfolio excess returns
    portfExRets = portfRets.subtract(riskFree[(riskFree.index >= portfRets.index[0])&(riskFree.index <= portfRets.index[-1])], axis = 0)

    # Calculate the portfolio Sharpe ratio
    SR = ((portfExRets + 1).apply(gmean, axis=0) - 1)/portfExRets.std()

    # Calculate the average turnover rate
    avgTurnover = np.mean(turnover[1:])

    #Print Sharpe ratio and Avg. turnover to the console
    # print("Elasped time is "+ str(end_time - start_time) + ' seconds')
    # print('Sharpe ratio: ', str(SR[0]))
    # print('Avg. turnover: ', str(avgTurnover))

    return (SR[0]), avgTurnover



def hyperparam_tune(turnover_weight: float, alpha: float, tau: float,
                       reg_lambda: float, transaction_cost_weight: float, rho:float):
    best_sharpe = -np.inf
    turn = []
    res = []
    best_param = None
    

    start_time = time.time()
    print("Calculating...")
    
    for j in range(len(tau)):
        sharpetemp, turntemp = back_test_main(turnover_weight, alpha, tau[j],
                    reg_lambda, transaction_cost_weight, rho)
        res.append([ tau[j], sharpetemp])
        if sharpetemp > best_sharpe:
            best_sharpe = sharpetemp
            best_param = tau[j]
            # turn.append(turntemp)
            
        

    end_time = time.time()
    print("Elasped time is "+ str(end_time - start_time) + ' seconds')

    # sharpe_ind, high = max(enumerate(best_sharpe), key=lambda x: x[1] )
    # turn_ind, low = min( enumerate(turn) , key=lambda x: x[1] )

    # print(f"Best Sharpe: {high}, Index: {combo[sharpe_ind]}")
    # print(f"Best Turoner: {low}, Index: {combo[turn_ind]}")

    print("Best Sharpe: ", best_sharpe)
    print("Best Tau: ", best_param)
    # print("Best Lambda: ", best_param[0])

    shp = [row[1] for row in res]
    # rh = [row[1] for row in res]
    tau = [row[0] for row in res]
    

    plt.figure(1)
    plt.plot(tau, shp, 'o')
    plt.title("Relationship Between Sharpe for RMVO vs Tau")
    plt.xlabel("Lambda")
    plt.ylabel("Sharpe")
    plt.show()

    # plt.figure(2)
    # plt.plot(rh, shp,'o')
    # plt.title("Relationship Between Sharpe for RMVO vs Rho ")
    # plt.xlabel("Rho")
    # plt.ylabel("Sharpe")
    # plt.show()


