import numpy as np
import pandas as pd
import pdb
import math
import sys

from sklearn.linear_model import Lasso as Lasso

###############################################
## Lasso s-map ################################
###############################################
def lasso_map(data, 
              obs,
              x,
              E = 3,
              theta=1,
              lasso_obj=Lasso(warm_start=True) ):
    '''
    This function takes a data array, a point and observables and
    returns a boolean vector, stating which variables are active in
    prediction and which are not.

    data  - lagged block - a numpy array, T by n
    obs   - observables - a numpy array, length T.
    x     - target - numpy array, length n.
    E     - embedding dimension, number of vars we want to use. Natural number
    theta - nonlinearity parameter. Real number

    '''
    ## Row diffs
    diffs = data - x

    ## Row distances from target
    dist = np.sqrt(np.sum( diffs * diffs, axis = 1 ) )
                   
    ## Exponential weights
    weights = np.exp( -theta * dist/np.mean(dist) )

    ## Reweight dataframe and observables
    data  = (data.T * weights).T
    obs = obs * weights

    ## Initialize the Lasso object and fit the data.
    lasso_obj.alpha = 1 ## Initialize to 1
    lasso_obj.fit(data, obs)
    active = lasso_obj.coef_ != 0 
    
    ## If we have the desired number of active components, just return
    if np.sum(active) == E:
        return active
    
    ## In this bit of code we ensure number of active variables for top
    ## is < E and number of active variables for bot is > E

    top = lasso_obj.alpha 
    bot = 0

    ## At the end of this loop
    while np.sum(active) > E:
        bot = top
        top = 2 * top
        lasso_obj.alpha = top
        lasso_obj.fit(data, obs)
        active = lasso_obj.coef_ != 0
        if np.sum(active) == E:
            return active

    ## Test, will remove later 
    # lasso_obj.alpha = bot
    # lasso_obj.fit(df,obs)
    # assert np.sum( active ) > n_active

    while True:
        mid = (top + bot)/2.
        lasso_obj.alpha = mid
        lasso_obj.fit(data, obs)
        active = lasso_obj.coef_ != 0
        
        if np.sum(active) == E:
            return active
        elif np.sum(active) > E:
            bot = mid
        else:
            top = mid
            
########################################################
## Get betas ###########################################
########################################################

def get_betas(var,
              df,
              lasso_obj=Lasso(warm_start=True)):
    '''
    df is assumed lagged, normalized and has a time column.

    gets betas (active variables) for prediction of var
    using cross validation at every time point
    '''

    ## Re-initialize the regularizing parameter to be one.
    lasso_obj.alpha = 1
      
    ## Observations are one time step ahead
    obs = pd.DataFrame(data = np.array(df[var + "_0"].shift(-1)),
                       index = df["time"],
                       columns = [var + "_p1"])
      
    # ## Find indices of rows that don't have NaNs
    # no_nans = ~np.logical_or(df.isnull().any(axis=1), obs.isnull() )

    # ## Keep only rows without NaNs
    # obs = obs[no_nans]
    # trim = df[no_nans]
    
    ## Preallocate. 
    betas = np.empty(trim.shape,
                     dtype=np.bool_)

    index = 0
    ## Iterate over points, find best predictors and store them
    for j , row in trim.iterrows():
        print( j )
        print( index )
        
        x = np.array(row)
        beta = lasso_map(trim,
                         obs,
                         x,
                         E=3,
                         theta=1,
                         lasso_obj=lasso_obj )
        
        betas[index, :] = beta
        index = index + 1
        if index % 10 == 0:
            print( "Cross validating. Variable " + str(var) + " using row " + str(index) + "."  )
        

    betas = pd.DataFrame(data=betas,
                         columns=df.columns,
                         index=trim["time"])
    
    betas.to_csv("data/active_" + var + ".csv",index_label="time")
    return betas

## TESTS??????????????????????????


