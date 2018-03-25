#!/usr/bin/python
import numpy as np
import pandas as pd
from math import exp as exp
import math
import pdb

import lasmap as lp

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

assert lp.simplex.generic(data, obs, x) == 1.


##############################################################
## Second test for generic method, using two and three nearest
## neighbours
##############################################################
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

## Just check we didn't screw up with the linear algebra.
assert abs(np.linalg.norm(data[0,:]-x) - 1.0) < 1e-14
assert abs(np.linalg.norm(data[1,:]-x) - 0.5) < 1e-14
assert abs(np.linalg.norm(data[2,:]-x) - 2.0) < 1e-14
assert abs(np.linalg.norm(data[3,:]-x) - 3.0) < 1e-14

## Using two nearest neighbours simplex the following calculation
## predicts the observable at x:
calc = (1 * exp(-1/0.5) + 2 * exp(-0.5/0.5) ) / ( exp(-1/0.5) + exp(-0.5/0.5) )

## Calling the function, we get:
func = lp.simplex.generic(data,
                          obs,
                          x,
                          num_nn=2)

## They need to be equal, at least up to numerical errors.
assert abs(func - calc) < 1e-14

## We want to comare our results with the rEDM package, so we save the
## data:
arr          = np.empty((5,4))
arr[0:4,0:3] = data
arr[0:4,  3] = obs
arr[4  ,0:3] = x
arr[4  ,  3] = calc
columns=["V1", "V2", "V3", "target" ]
df = pd.DataFrame(columns=columns,
                  data=arr)
df.to_csv("data/test_data_2NN.csv",
          index=False)

## Similarly, using three nearest neighbours:
calc = (1 * exp(-1/0.5) + 2 * exp(-0.5/0.5) + 3 * exp(-2./0.5) ) / ( exp(-1/0.5) + exp(-0.5/0.5) + exp(-2./0.5) )
func = ls.simplex.generic(data,
                          obs,
                          x,
                          num_nn=3)

## They better be equal...
assert abs(func - calc) < 1e-14

## Save this for rEDM, only changing the result.
df.at[4,"target"]=calc

df.to_csv("data/test_data_3NN.csv",
          index=False)

#####################################
## Test generic_sets
#####################################
np.random.seed(19)

## Create a dataframe and populate it with fake data.
lib = [1,2,5,6,8]
pred = [9,11,13,14,16]
columns = ["V1", "V2", "V3", "target" ]
predictors = ["V1", "V2", "V3"]
target = "target"
df = pd.DataFrame(index=np.ravel([lib,pred]),
                  columns=columns,
                  data=np.random.normal(size=(len(lib)+len(pred),len(columns)))) 

calc = ls.simplex.generic_sets(df.reindex(labels=lib, axis="index"),## df.reindex(labels=lib, axis="index"),
                               df.reindex(labels=pred, axis="index"),
                               target,
                               predictors)

## Set this data in the dataframe
df.at[pred,"target"] = calc["pred"]

# Save...
df.to_csv("data/test_data_generic_sets.csv",
          index=False)

######################################################
## Test the univariate method
######################################################
np.random.seed(19)

## Generate fake data
n = 12
df = pd.DataFrame(index=np.arange(n)*10,
                  columns=["A"],
                  data=np.random.randn(n))
preds = ls.simplex.univariate(df,
                              2,
                              tp=2)
preds["truth"] = df["A"]
preds.to_csv("data/test_data_univariate.csv",
             index=True,
             index_label="time",
             na_rep = "NaN")

