import numpy as np
import pandas as pd
import os

def make_simple_data():

    '''Data with known distances between data points

    '''

    x = np.ones(3)
    obs = np.array([1.,2.,3.,4.])
    data = np.empty( (4,3) )

    ## So first line is distance 1 from x
    v = np.array( [2,-1,3] )
    v = v / np.linalg.norm(v)
    data[0,:] = x + v

    ## So second line is distance 1/2 from x
    v = np.array( [-1,0,3])
    v = v / np.linalg.norm(v) / 2.
    data[1,:] = x + v

    ## So third line is distance 2 from x
    v = np.array( [1,6,2])
    v = v / np.linalg.norm(v) * 2.
    data[2,:] = x + v
    
    ## So fourth line is distance 3 from x
    v = np.array( [0,3,-4])
    v = v / np.linalg.norm(v) * 3.
    data[3,:] = x + v
    
    ## We want to comare our results with the rEDM package, so we save the
    ## data:
    arr          = np.empty((5,4))
    arr[0:4,0:3] = data
    arr[0:4,  3] = obs
    arr[4  ,0:3] = x
    columns = ["V1", "V2", "V3", "target" ]
    df = pd.DataFrame(columns=columns,
                      data=arr)

    df.to_csv(filename,
              index=False)

    # return {"data" : data, "x": x, "obs": obs }
