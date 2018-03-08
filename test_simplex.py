#!/usr/bin/python
import numpy as np
import pandas as pd
from math import exp as exp
import math
import pdb

import simplex

############################################
## First Test ##############################
############################################

## First test. Very simple.
data = np.array( [
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 2, 2, 2],
    [2, 0, 2, 2],
    [1, 1, 1, 0],
    [0, 0, 0, 0],
] )

obs = np.array( [ 1, 1, 1, 100, 200, 1, 1 ] )

x = np.array( [0.5, 0.5, 0.5, 0.5] )

assert simplex.generic(data, obs, x) == 1.


############################################
## Second test for generic method ##########
############################################
x = np.ones(3)
obs = np.array([1.,2.,3.,4.])
data = np.empty( (4,3) )

## So first line is distance 1 from x
v = np.array( [2,-1,3] )
v = v / np.linalg.norm(v)
data[0,:] = x + v

v = np.array( [-1,0,3])
v = v / np.linalg.norm(v) / 2.
data[1,:] = x + v

v = np.array( [1,6,2])
v = v / np.linalg.norm(v) * 2.
data[2,:] = x + v

v = np.array( [0,3,-4])
v = v / np.linalg.norm(v) * 3.
data[3,:] = x + v


assert abs(np.linalg.norm(data[0,:]-x) - 1.0) < 1e-14
assert abs(np.linalg.norm(data[1,:]-x) - 0.5) < 1e-14
assert abs(np.linalg.norm(data[2,:]-x) - 2.0) < 1e-14
assert abs(np.linalg.norm(data[3,:]-x) - 3.0) < 1e-14


## Using two nearest neighbours, the analytic calculation gives:
calc = (1 * exp(-1/0.5) + 2 * exp(-0.5/0.5) ) / ( exp(-1/0.5) + exp(-0.5/0.5) )

## The function call gives:
func = simplex.generic(data,
                       obs,
                       x,
                       num_nn=2)

assert abs(func - calc) < 1e-14

## Save the data for R:
arr=np.empty((5,4))
arr[0:4,0:3]= data
arr[0:4,  3]= obs
arr[4  ,0:3]= x
arr[4  ,  3]= calc
df = pd.DataFrame(columns=["V1", "V2", "V3", "target" ],
                  data=arr)
df.to_csv("data/test_data_2NN.csv",
          index=False)

## Using three nearest neighbours:
calc = (1 * exp(-1/0.5) + 2 * exp(-0.5/0.5) + 3 * exp(-2./0.5) ) / ( exp(-1/0.5) + exp(-0.5/0.5) + exp(-2./0.5) )
func = simplex.generic(data,
                       obs,
                       x,
                       num_nn=3)

assert abs(func - calc) < 1e-14
df.iloc[4,3]=calc

df.to_csv("data/test_data_3NN.csv",
          index=False)


# ## Run and save using fake data
# data = pd.read_csv("data/test_data.csv",
#                  index_col="time")

# preds = simplex.univariate(data,2)

# df = data.merge(preds,
#                 how="right",
#                 right_index=True,
#                 left_index=True)
# df.to_csv("data/test_results.csv",
#           na_rep = "nan",
#           mode = "a" )

