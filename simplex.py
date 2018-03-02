import numpy as np
import pandas as pd
import pdb

import helpers
## Block is the method that actually performs simplex projection. It
## does so for one point only. All other methods are wrappers around
## this method --- they manage generating block structure etc.

def generic(data,
            obs,
            x,
            num_nn=0):
    
    '''
    A generic method to perform simplex projection. Takes numpy arrays.
    
    data is a numpy array, T by E.
    
    obs is a numpy array of length T - one observable for every time
    point.

    x is a numpy array of length E.

    '''

    
    if num_nn == 0:
        num_nn = len(x)+1

    ## Row distances from x. Calculate norm by summing over axis #1.
    dist = np.linalg.norm(data - x, axis=1)

    ## Get the num_nn points with least distance. 
    ind = np.argsort(dist)[0:num_nn]

    ## Prediction is a weighted mean of nearest neighbours
    ## observations.
    obs  = obs[ind]
    w    = np.exp(-dist[ind])
    
    pred = np.sum( w * obs ) / np.sum( w )

    return pred

def univariate(lib,
               E,
               tp=1):
    '''Perform univariate prediction with cross validation. Returns
    dataframe of predictions and observations.  

    We always assume the index column is time.

    '''

    ## We verify data frame only has one variable. Consequently, this
    ## is the variable name and the one that is lagged.
    assert len(lib.columns == 1)
    varName = lib.columns[0]
    
    ## Create lagged block
    block = helpers.lag(lib, E)

    ## Add an observation ( "_p1" ) column.
    block = helpers.extend_obs(block, varName, tp=tp)

    ## Keep the observables as a separate data frame...
    obs = block[varName + "_p" + str(tp)]

    ## ... and remove from the block
    block = block.drop([varName + "_p1"],axis=1)

    # Preallocate the predicted data frame. This line creates a data
    # frame with NaN values with the size of obs and indices of obs:
    ret = pd.DataFrame(index=obs.index,
                         columns=["pred"],
                         data=np.full(len(obs), np.nan) )

    for row_index, row_data in block.iterrows():

        ## Remove current row from the data fed to the generic method.
        tmp_block = block.drop(row_index)
        tmp_obs   = obs.drop(row_index)
        
        ## Indexing by time stamp, so use loc and *not* iloc.
        ret.loc[row_index] = generic(tmp_block.values,
                                     tmp_obs.values,
                                     row_data.values)
        
    ## Extend the returned object
    ret["obs"] = obs
    return ret
    
        
