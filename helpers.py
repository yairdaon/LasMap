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

        
## Got this function online I think, no tests.
def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

