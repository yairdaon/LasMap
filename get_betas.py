#!/usr/bin/python
import numpy as np
import pandas as pd
import pdb
import math
import sys
import helpers

from sklearn.linear_model import Lasso as Lasso

df = pd.read_csv( "data/huisman.csv" )
df = df.iloc[100:200]
# pdb.set_trace()
df = helpers.normalize(df,time="time")
df = helpers.add_noise(df,
                       sig=math.sqrt(0.1),
                       time="time")
df.to_csv("data/noisy_huisman.csv",
              index=False)

## The dictionary of lags
lags = {}
lags["N1"] = [0,1,2]
lags["N2"] = [0,1,2]
lags["N3"] = [0,1,2]
lags["N4"] = [0,1,2]
lags["N5"] = [0,1,2]

## Lag coordinates
df = helpers.lag(df,lags, time="time")

lasso_obj = Lasso(warm_start=True)

for var in ["N1", "N2", "N3", "N4", "N5"]:
    
    ## Observations are one time step ahead
    obs = df[var + "_0"].shift(-1)
    
    ## Find indices of rows that have NaN
    no_nans = ~np.logical_or(df.isnull().any(axis=1), obs.isnull() )

    obs = obs[no_nans]
    trim = df[no_nans]
    
    ## Preallocate
    betas = np.empty(trim.shape,
                     dtype=np.bool_)

    index = 0
    ## Iterate over points, find best predictors and store them
    for _ , row in trim.iterrows():
        x = np.array(row)
        beta = helpers.lasso_map(trim,
                                 obs,
                                 x,
                                 E=3,
                                 theta=1,
                                 lasso=lasso_obj )
        
        betas[index, :] = beta
        index = index + 1
        if index % 10 == 0:
            print( "Cross validating using row " + str(index) + "."  )
        

    betas = pd.DataFrame(data=betas,
                         columns=df.columns,
                         index=trim["time"])
    
    betas.to_csv("data/active_" + var + ".csv",index_label="time")
            
