import numpy as np
import pandas as pd
import pdb
import warnings

import helpers

'''Generic is the method that actually performs simplex
projection. It does so for one point only. All other methods are
wrappers around this method --- they manage generating block structure
etc.

We ***always*** assume there is no "time" column - if there are time
stamps, these are always assumed to be in the index of the
dataframe. If there is a time/data/whatever column (that is not the
index) - you can expect weird behaviour.

'''

def univariate(lib,
               E,
               tp=1):
    
    '''Performs univariate prediction with cross validation. Returns
    dataframe of predictions and observations.

    We always assume the index column is time.

    tp is ***time-steps***. That is to say, if the index (time) set is
    [10,20,30,40,50,...], then tp==1 means that for index (time) 10 we
    try to predict the value at index (time) 20. Thus, it is implicity
    assumed that observations are equally spaced and ordered.

    '''

    ## We verify data frame only has one variable. Consequently, this
    ## is the variable name and the one that is lagged.
    assert len(lib.columns) == 1
    varName = lib.columns[0]

    ## Prediction of the past and present is not allowed. It does not
    ## make much sense either.
    assert tp > 0
    
    target = varName + "_p" + str(tp)
    
    ## Create lagged block
    block = helpers.lag(lib, E)

    ## Add an observation ( "_p1" ) column.
    block = helpers.extend_obs(block, varName, tp=tp)

    ## Keep the observables as a separate data frame...
    obs = block[target]
    
    ## ... and remove from the block
    block = block.drop([target],axis=1)

    predictors = block.columns
    
    ## Preallocate the predicted data frame. We will later need to
    ## shift the data so that obs and pred are aligned with it.
    ret = pd.DataFrame(index=obs.index,
                       columns=["pred"],
                       data=np.full(len(obs), np.nan) )
    
    for row_index, row_data in block.iterrows():
        
        ## Remove current row from the data fed to the generic method.
        tmp_block = block.drop(row_index)
        tmp_obs   = obs.drop(row_index)

        ## Indexing by time stamp, so use at and *not* iat.
        ret.at[row_index, "pred"] = generic(tmp_block.values,
                                            tmp_obs.values,
                                            row_data.values,
                                            num_nn=E+1)

    ## As promised above, we shift the data so that pred is aligned
    ## with the time index.
    ret = ret.shift(tp)

    ## Set the observations to be the truth.
    ret["obs"] = lib[varName]

    return ret
        
def generic_sets(lib_set,
                 pred_set,
                 target,
                 predictors,
                 num_nn=0):
    '''We get two data frames. Use lib_set to make predictions on
    pred_set. We try to guess target column, given predictors columns.

    '''

    if not isinstance(lib_set, pd.DataFrame):
        raise ValueError( """lib_set is not a DataFrame. Possible reason: slicing a DataFrame by
        (e.g.) df.loc[4] returns a Series. Slicing by df.loc[4:4] returns a
        DataFrame.""" )

    if not isinstance(pred_set, pd.DataFrame):
        raise ValueError( """pred_set is not a DataFrame. Possible reason: slicing a DataFrame
        (e.g.) by df.loc[4] returns a Series. Slicing by df.loc[4:4] returns
        a DataFrame.""" )
    
    ret = pd.DataFrame(index=pred_set.index,
                       columns=["obs"],
                       data=pred_set["target"].values )
    ret["pred"] = np.full(len(ret), np.nan)

    data  = lib_set[ predictors ] 
    obs   = lib_set[ target ] ## obs for the generic method
    block = pred_set[ predictors ]
    
    for row_index, row_data in block.iterrows():
        
        ret.at[row_index, "pred"] = generic(data.values,
                                            obs.values,
                                            row_data.values,
                                            num_nn=num_nn)
        
    return ret


def generic(data,
            obs,
            x,
            num_nn=0):
    
    '''A generic method to perform simplex projection. Takes numpy arrays
       
    data is a numpy array, T by E.
    
    obs is a numpy array of length T - one observable for every time
    point.

    x is a numpy array of length E.
    
    Below, 1e-6 is hard coded to match the definition of min_weight
    from Sugihara's original C++ code, see
    https://github.com/ha0ye/rEDM/src/forecastmachine.cpp

    '''

    if not isinstance(data, np.ndarray):
        warnings.warn( """Variable data in simplex.generic method is not a numpy array
        (probably a DataFrame). Expect unexpected behaviour.""")

    if not isinstance(obs, np.ndarray):
        warnings.warn( """Variable obs in simplex.generic method is not a numpy array
        (probably a DataFrame). Expect unexpected behaviour.""")
    
    if not isinstance(x, np.ndarray):
        warnings.warn( """Variable x in simplex.generic method is not a numpy array (probably
        a DataFrame). Expect unexpected behaviour.""")

    if np.isnan(x).any():
        warnings.warn("Data point contains NaN values. Returning NaN.")
        return np.nan
    
    if num_nn == 0:
        num_nn = len(x)+1

    ## We keep the rows that are not nan in both arrays.
    nan_obs  = np.isnan(obs) 
    nan_data = np.isnan(data).any(axis=1) 
    keep     = ~np.logical_or(nan_obs, nan_data) 

    ## If we have less valid data points (with observations) than the
    ## required number of nearest neighbours, we cannot make any
    ## prediction and return a NaN.
    if np.sum( keep ) < num_nn:
        warnings.warn("Required NN " + str(num_nn) + " but only " + np.sum(keep) + " valid data points available. Return NaN." )
        return np.nan

    ## After this, data should be valid and we can actually make a prediction!
    obs = obs[keep]
    data = data[keep]

    ## Row distances from x. Calculate norm by summing over axis #1
    ## which means Sum_j a_ij, in standard linear algebraic terms.
    dist = np.linalg.norm(data - x, axis=1)

    ## Get the num_nn points with least distance. 
    ind = np.argsort(dist)[0:num_nn]
   
    ## Prediction is a weighted mean of nearest neighbours'
    ## observations:
    obs  = obs[ind] ## NN observations
    w = np.exp(-dist[ind] / np.min(dist) ) ## NN weights

    ## See docs above.
    w = np.maximum( w, 1e-6 ) 

    ## Simplex predicts as follows:
    pred = np.sum( w * obs ) / np.sum( w )
    
    return pred

