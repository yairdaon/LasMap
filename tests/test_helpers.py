#!/usr/bin/python
import numpy as np
import pandas as pd
import pdb
import sys

from lasmap import helpers

def test_normalize():
        arr = np.array([ [1,3,4],
                         [4,3,6],
                         [2,4,5],
                         [8,9,0],
                         [2,5,6],
                         [1,7,8] ], dtype = np.float64 )
        df = pd.DataFrame(index=range( arr.shape[0]),
                          columns=[ "x", "y", "z" ],
                          data=arr)

        normalized, scaler = helpers.normalize(df, return_scaler=True)
        assert np.all( np.abs(normalized.mean()[1:]) < 1e-15 ) 

        unnormalized = normalized * np.sqrt(scaler.var_) + scaler.mean_
        assert np.max( np.abs(df.values-unnormalized.values) ) < 1e-15
        

## TODO: Maybe I should actually make a test of this???
def test_lag():
    arr = np.array([ [1,3,4],
                     [4,3,6],
                     [2,4,5],
                     [8,9,0],
                     [2,5,6],
                     [1,7,8] ], dtype = np.float64 )
    df = pd.DataFrame(index=range( arr.shape[0]),
                      columns=[ "x", "y", "z" ],
                      data=arr)
       
    ## Lag using a dictionary
    lagged = helpers.lag(df, { "x" : [0,2,3], "y" : [0,3], "z" : [0,1] } )
    print( lagged )

    ## Lag using an integer
    lagged = helpers.lag(df, 3)
    print( lagged )


