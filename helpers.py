import numpy as np
import pandas as pd
import pdb
import math

from sklearn.linear_model import Lasso as Lasso
from sklearn import preprocessing as prep 
    
##############################################
## Add observations column ###################
##############################################
def extend_obs(df,
               varName,
               tp=1):
    '''
    Extends data frame with a column of future (tp) observations.

    varName is assumed given as X, y, chlA etc. and *not* X_0, y_0,
    chlA_0.
    
    df is assumed lagged (hence columns are X_0, X_1, y_0, y_2,
    chlA_0, chlA_1, chl_A_2 etc.

    '''
    assert tp > 0

    ## Shift the zero lagged variable
    obs = df[varName + "_0"].shift(-tp) 

    ## Add / append the shifted variable to the data frame
    df[varName + "_p" + str(tp)] = obs
    return df

##############################################
## Lagging function ##########################
##############################################

## Lags df according to lags. Takes a time column and makes it the
## index.
def lag(df,
        lags,
        time=None):

    if time is not None:
        df = df.set_index(time)
   
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
        
        
## Got this function online I think, no tests.
def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

