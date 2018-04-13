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
    assert "x_0" in lagged.columns
    assert "x_2" in lagged.columns
    assert "x_3" in lagged.columns
    assert "y_3" in lagged.columns
    assert "z_0" in lagged.columns
    assert "z_1" in lagged.columns
    assert len(lagged.columns) == 7
    
    np.testing.assert_equal( lagged["x_0"], np.array([                         1, 4, 2, 8, 2, 1 ] ) )
    np.testing.assert_equal( lagged["x_2"], np.array([ np.nan, np.nan,         1, 4, 2, 8       ] ) )
    np.testing.assert_equal( lagged["x_3"], np.array([ np.nan, np.nan, np.nan, 1, 4, 2          ] ) )
    np.testing.assert_equal( lagged["y_0"], np.array([                         3, 3, 4, 9, 5, 7 ] ) )
    np.testing.assert_equal( lagged["y_3"], np.array([ np.nan, np.nan, np.nan, 3, 3, 4          ] ) )
    np.testing.assert_equal( lagged["z_0"], np.array([                         4, 6, 5, 0, 6, 8 ] ) )
    np.testing.assert_equal( lagged["z_1"], np.array([ np.nan,                 4, 6, 5, 0, 6    ] ) )
    



    
    ## Lag using an integer
    lagged = helpers.lag(df, 3)
    assert "x_0" in lagged.columns
    assert "x_1" in lagged.columns
    assert "x_2" in lagged.columns
    assert "y_0" in lagged.columns
    assert "y_1" in lagged.columns
    assert "y_2" in lagged.columns
    assert "z_0" in lagged.columns
    assert "z_1" in lagged.columns
    assert "z_2" in lagged.columns
    assert len(lagged.columns) == 9

    ## Test some and not all cuz im lazy
    np.testing.assert_equal( lagged["x_0"], np.array([                         1, 4, 2, 8, 2, 1 ] ) )
    np.testing.assert_equal( lagged["x_2"], np.array([ np.nan, np.nan,         1, 4, 2, 8       ] ) )
    np.testing.assert_equal( lagged["y_0"], np.array([                         3, 3, 4, 9, 5, 7 ] ) )
    np.testing.assert_equal( lagged["y_2"], np.array([ np.nan, np.nan,         3, 3, 4, 9       ] ) )
    np.testing.assert_equal( lagged["z_0"], np.array([                         4, 6, 5, 0, 6, 8 ] ) )
    np.testing.assert_equal( lagged["z_1"], np.array([ np.nan,                 4, 6, 5, 0, 6    ] ) ) 
    
