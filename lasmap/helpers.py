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
    
    df is assumed lagged hence columns are (e.g.) X_0, X_1, y_0, y_2,
    chlA_0, chlA_1, chl_A_2.

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
        lags):
   
    ## If lags is int, we lag all variables 0,...,lags-1. Here we
    ## create the lagging dictionary.
    if isinstance(lags, int):
        dic = {}
        for col in df.columns:
            dic[col] = range(lags)
        lags = dic
    
    lagged = pd.DataFrame(index=df.index)
    for var, lag_list in lags.iteritems():
        for lag in lag_list:
            col = df[var].shift(lag)
            lagged[ var + "_" + str(lag) ] = col
            
    return lagged

##############################################
## Add noise #################################
##############################################
def add_noise(df,
              sig=.1):
    
    '''Add centered iid Gaussian noise with standard deviation sig to the
    dataframe df. Note that we implicitly assume data is normalized
    (hence the uniform noise amplitude, which would be unfit had the
    data not been normalized) and that there is no time column - time is
    assumed to be the index of the dataframe.

    '''
    return df + np.random.normal(0,sig,df.shape)
        
##############################################
## Normalize #################################
##############################################
def normalize(df,
              return_scaler = False):

    '''Normalize a datafram and possibly return the scaling function. If
    return_scaler == True, then this function also returns scaler such
    that scaler(df) is normalized.

    '''
    ## Get the routine that normalizes given df to zero mean unit
    ## variance
    scaler = prep.StandardScaler()

    
    ## Rescale
    normalized = pd.DataFrame(index=df.index,
                              columns=df.columns,
                              data=scaler.fit_transform(df.values))
        
    if return_scaler:
        return normalized, scaler
    return normalized            

## Got this function online I think, no tests.
def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)