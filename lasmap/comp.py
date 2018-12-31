import numpy as np
import pandas as pd
import pdb
import math
from sklearn.linear_model import Lasso as Lasso

from lasmap import helpers

###############################################
## Lasso s-map ################################
###############################################
def lasso_map(data, 
              obs,
              x,
              E,
              theta=1,
              lasso_obj=Lasso(warm_start=True) ):
    '''This function takes a numpy array, a point and observables and
    returns a boolean vector, stating which variables are active in
    prediction of the observable for the point and which are not.

    data  - (lagged) block - a pandas data frame, T by n
    obs   - observables - a numpy array, length T.
    x     - target - numpy array, length n.
    E     - embedding dimension, number of active variables we want to use. Natural number
    theta - nonlinearity parameter. Real number

    '''

    ## Row distances from x
    dist = np.linalg.norm(data - x, axis=1)
                   
    ## Exponential weights
    weights = np.exp( -theta * dist/np.mean(dist) )

    ## Reweight dataframe and observables
    data  = (data.T * weights).T ## TODO: maybe optimize this using einsum
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
              E=3,
              theta=1,
              tp=1,
              lasso_obj=Lasso(warm_start=True)):
    
    '''df is assumed lagged and normalized. Any time/date/etc. column is
    always assumed to be the index.

    gets betas (active variables) for prediction of var (tp time steps
    ahead) using cross validation at every time point

    '''

    ## Re-initialize the regularizing parameter to be one.
    lasso_obj.alpha = 1
      
    ## Observations are tp time steps ahead
    obs = helpers.get_obs(df,var,tp)

    ## Preallocate. According to StackOverflow, this is inferior
    ## (speedwise) to holding rows in a list, then creating the
    ## dataframe. This can be optimized but for now - fuckit. See
    ## https://stackoverflow.com/questions/18771963/pandas-efficient-dataframe-set-row
    betas = pd.DataFrame(data = np.empty(df.shape, dtype=np.bool_),
                         columns = df.columns,
                         index = df.index)

    ## Remove all rows with nans, lagged and future observations
    df, obs = helpers.remove_nan_rows(df, obs)
   
    ## Iterate over points, find best predictors and store them
    for ind, row in df.iterrows():

        beta = lasso_map(df.drop(ind, axis=0).values,
                         obs.drop(ind, axis=0).values,
                         row.values,
                         E,
                         theta,
                         lasso_obj )
        
        betas.at[ind] = beta
        
        print( "Cross validating. Variable " + str(var) + " using time stamp " + str(ind) + "."  )

    return betas

