#!/usr/bin/python
import numpy as np
import pandas as pd
import pdb
import math

from sklearn.linear_model import Lasso as Lasso
from sklearn import preprocessing as prep 

##############################################
## Add noise #################################
##############################################
def add_noise(df,
              sig,
              time=None):
    if time is not None:
        ts = df[time]

    noise = np.random.normal(0,sig,df.shape)
    df = df + noise
    
    if time is not None:
        df[time] = ts

    return df
        
##############################################
## Normalize #################################
##############################################
def normalize(df, time=None):
    if time is not None:

        ## For some reason casting helps, I have no clue why. Test
        ## passes wihout casting though.
        ts = np.array(df[time], dtype=np.int32)
   
    ## Normalize df to zero mean unit variance
    names = df.columns
    df = prep.scale(df,axis=0)
    df = pd.DataFrame(data=df,
                      columns = names)

    if time is not None:
        df[time] = ts
    
    return df

## Test
if __name__ == "__main__":

        arr = np.array([ [1,1,3,4],
                         [2,4,3,6],
                         [3,2,4,5],
                         [4,8,9,0],
                         [5,2,5,6],
                         [6,1,7,8] ], dtype = np.float64 )
        df = pd.DataFrame( arr )
        df.columns = ["time", "x", "y", "z" ]
        normalized = normalize(df,time="time")
        assert np.all( np.abs(normalized.mean()[1:]) < 1e-14 ) 

        
##############################################
## Lagging function ##########################
##############################################

## Lags df according to lags. Maintains the time column.
def lag(df,
        lags,
        time=None):

    if time is None:
        lagged = pd.DataFrame()
    else:
        # pdb.set_trace()
        ts = np.array(df[time],dtype=np.int32)
        lagged = pd.DataFrame(data=ts,
                              columns=[time])
   
    ## If lags is int, we lag all variables 0,...,lags-1. Here we
    ## create the lagging dictionary.
    if isinstance( lags, int ):
        dic = {}
        for col in df.columns:
            dic[col] = range(lags)
        lags = dic

    for var, lag_list in lags.iteritems():
        if var == time:
            continue
        for lag in lag_list:
            col = df[var].shift(lag)
            lagged[ var + "_" + str(lag) ] = col
            
    return lagged

## Test
if __name__ == "__main__":
    arr = np.array([ [1,1,3,4],
                     [2,4,3,6],
                     [3,2,4,5],
                     [4,8,9,0],
                     [5,2,5,6],
                     [6,1,7,8] ], dtype = np.float64 )
    df = pd.DataFrame( arr )
    df.columns = ["time", "x", "y", "z" ]

    ## Lag using a dictionary
    lagged = lag(df,
                 { "x" : [0,2,3], "y" : [0,3], "z" : [0,1] },
                 time = "time" )
    print( lagged )

    ## Lag using an integer
    lagged = lag(df, 3, time="time")
    print( lagged )


###############################################
## Lasso s-map ################################
###############################################
def lasso_map(df,
              obs,
              x,
              E = 3,
              theta=1,
              lasso=Lasso(warm_start=True) ):
    '''
    df    - lagged block. T by n
    obs   - observables. T
    x     - target. n
    E     - embedding dimension, number of vars we want to use. Natural number
    theta - nonlinearity parameter. Real number
    '''
    ## Row diffs
    diffs = df - x

    ## Row distances from target
    dist = np.sqrt(np.sum( diffs * diffs, axis = 1 ) )

    non_zero = dist > 1e-9
    dist     = dist[ non_zero ]
    df       = df[ non_zero ]
    obs      = obs[ non_zero ]
                   
    ## Exponential weights
    weights = np.exp( -theta * dist/np.mean(dist) )

    ## Reweight dataframe and observables
    df  = (df.T * weights).T
    obs = obs * weights

    ## Initialize the Lasso object and fit the data.
    lasso.alpha = 1 ## Initialize to 1
    lasso.fit(df, obs)
    active = lasso.coef_ != 0 
    
    ## If we have the desired number of active components, just return
    if np.sum(active) == E:
        return active
    
    ## In this bit of code we ensure number of active variables for top
    ## is < E and number of active variables for bot is > E

    top = lasso.alpha 
    bot = 0

    ## At the end of this loop
    while np.sum(active) > E:
        bot = top
        top = 2 * top
        lasso.alpha = top
        lasso.fit(df, obs)
        active = lasso.coef_ != 0
        if np.sum(active) == E:
            return active

    ## Test, will remove later 
    # lasso.alpha = bot
    # lasso.fit(df,obs)
    # assert np.sum( active ) > n_active

    while True:
        mid = (top + bot)/2.
        lasso.alpha = mid
        lasso.fit(df,obs)
        active = lasso.coef_ != 0
        
        if np.sum(active) == E:
            return active
        elif np.sum(active) > E:
            bot = mid
        else:
            top = mid
            
if __name__ == "__main__":

    ## Test many a times we actually get the correct number of
    ## non-zero coefficients
    for i in range( 1500 ):
        ## Design matrix (data frame)
        # X = np.array([ [1,3,4,6,5,6],
        #                [4,3,6,9,0,3],
        #                [2,4,5,3,1,8],
        #                [8,9,0,8,0,3],
        #                [2,5,6,1,2,7],
        #                [1,7,8,3,4,5] ], dtype = np.float64 )
        X = np.random.randn( 210, 12 )
        X = pd.DataFrame( X )
        # X.columns = ["a", "b", "c", "d", "e", "f"]

        ## True beta and E
        beta = np.random.randn( X.shape[1] ) * np.random.choice([0, 1], size=(X.shape[1],), p=[1./3, 2./3])
        E = np.random.randint(low=2,high=5)

        ## Noisy observations with intercept added
        y = np.einsum( "ij, j -> i", X, beta ) + 2 
        y = y + np.random.randn( len(y) ) * 2
        
        ## A random vector in R^d
        x = np.random.randn( X.shape[1] )

        beta_hat = lasso_map(X,
                             y,
                             x,
                             E=E,
                             theta=np.random.uniform(0,5) )

        ## Did we get what we wanted?
        assert np.sum(beta_hat) == E
