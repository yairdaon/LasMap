import numpy as np
import pandas as pd
## Block is the method that actually performs simplex projection. It
## does so for one point only. All other methods are wrappers around
## this method --- they manage generating block structure etc.

def generic(data,
            obs,
            x,
            num_nn=len(x)+1):
    
    '''
    A generic method to perform simplex projection. Takes numpy arrays.
    
    data is a numpy array, T by E.
    
    obs is a numpy array of length T - one observable for every time
    point.

    x is a numpy array of length E.

    '''
    
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
    
def univariate(lib, E):
    ''' 
    Perform univariate prediction with cross validation.
    
    We always assume the index column is time.
    '''

    ## We verify data frame only has one variable. Consequently, this
    ## is the variable name and the one that is lagged.
    assert len(lib.columns == 1)
    varName = lib.columns[0]
    
    ## Create lagged block
    block = helpers.lag(lib, E)

    ## Add an observation ( "_p1" ) column.
    block = helpers.extend_obs(block, varName)

    ## Keep the observables as a separate data frame...
    obs = block[varName + "_p1"]

    ## ... and remove from the block
    block = block.drop([varName + "_p1"],axis=1)

    # preallocate the predicted data frame:
    preds = pd.DataFrame(index=obs.index,
                         columns=obs.columns)

    for row_index, row_data in block.iterrows():

        ## Remove current row from the data fed to the generic method.
        tmp_block = block.drop(row_index)
        tmp_obs   = obs.drop(row_index)
        
        ## Indexing by time stamp, so use loc and *not* iloc.
        preds.loc[row_index] = simplex(tmp_block,
                                       tmp_obs,
                                       row_data)
        

    return preds
