import numpy as np
import pandas as pd

def block(data,
          obs,
          x,
          num_nn=len(x)+1):
    
    '''
    a generic method to perform simplex projection.
    
    data is a numpy array, T by E.
    
    obs is a numpy array of length T - one observable for every time
    point.

    x is a numpy array of length E.

    '''
    
    ## Row diffs
    diff = data - x

    ## Row distances from x
    dist = np.linalg.norm(diff, axis=1)

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
               pred=None,
               time=None):
    ''' 
    does univariate prediction, uses cross validation by default.
    '''

    lib_block = helpers.lag(df, E, time)
    oos_block = 

def join(df,
         target):
    
    pass

def seperate(df,
             obs):
    
    pass
