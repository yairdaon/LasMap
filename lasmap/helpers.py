import numpy as np
import pandas as pd
import pdb
import math

from sklearn.linear_model import Lasso as Lasso
from sklearn import preprocessing as prep

def remove_nan_rows(data, obs=None):

    ind = no_nan_rows(data, obs)
   
    if isinstance(data, np.ndarray):
        no_nan_data = data[ind, :]
    elif isinstance(data, pd.DataFrame):
        no_nan_data = data.iloc[ind]
    else:
        raise ValueError( "data has to be either numpy array or pandas data frame." )

    if obs is None:
        return no_nan_data
    else:
        if isinstance(data, np.ndarray):
            return no_nan_data, obs[ind]
        elif isinstance(data, pd.DataFrame):
            return no_nan_data, obs.iloc[ind]
        else:
            raise ValueError( "obs has to be either numpy array or pandas data frame." )
        
def no_nan_rows(data, obs=None):

    '''data is either a 2D numpy array or a pandas data frame.'''
    
    ## Find indices of rows that have NaNs
    data_nan_rows = np.isnan(data).any(axis=1)
    
    if obs is None:
        nan_rows = data_nan_rows
    else:
        
        ## No np.ravel - obs is a Series and shape == (n,).
        obs_nan_rows = np.isnan(obs)

        ## Rows that have NaN in either data or obs.
        nan_rows     = np.logical_or( data_nan_rows, obs_nan_rows )
        
    no_nan_rows = ~nan_rows
    
    ## Get the indices of rows where data (and obs) are nan free. We
    ## cast to a tuple because a numpy array cannot be used to index a
    ## numpy array.
    return np.flatnonzero( no_nan_rows )
        
##############################################
## Create observations column ################
##############################################
def get_obs(df,
            varName,
            tp=1):
    '''Get observation (i.e. future, "_p1") Series object from an existing
    data frame. We use a Series object rather than a DataFrame because
    a series has shape (n,) and ***not*** (n,1)

    varName is assumed given as X, y, chlA etc. and *not* X_0, y_0,
    chlA_0.
    
    df is assumed lagged hence columns are (e.g.) X_0, X_1, y_0, y_2,
    chlA_0, chlA_1, chl_A_2.

    '''
    assert tp > 0

    return pd.Series(data = df[varName + "_0"].shift(-tp).values,
                     index = df.index )

##############################################
## Lagging function ##########################
##############################################
def lag(df,
        lags):

    '''Lags df according to lags.

    '''
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
