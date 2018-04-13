#!/usr/bin/python
import numpy as np
import pandas as pd
import pdb
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

import helpers
import simplex
import comp

## Read entire data
df = pd.read_csv("~/lasmap/huisman/huisman.csv",
                 index_col="time")

df["N3+N5"] = df["N3"] + df["N5"]
df["N2+N4"] = df["N2"] + df["N4"]

## Normalize
df = helpers.normalize(df)

## Restrict data, because we don't want to use ALL of it. Huisman data
## goes from 10 to 6000, in increments of 10. Hence length of data
## frame is 600. In this region, N2 is approximately constant.
cut = df.loc[3550:3790]

################################################################
## Plot all time series on all of their time span.
################################################################
for ind, col in enumerate(df.columns):

    series = df[col]
    time   = df.index

    plt.subplot( len(df.columns), 1, ind+1 )
    plt.plot(time, series)
    plt.ylabel( col )
    
plt.savefig("/home/yair/lasmap/huisman/pix/python/full_time_series.png")
plt.close()

################################################################
## Plot all time series on the restricted region, where N2 is
## approximately constant
################################################################
for ind, col in enumerate(cut.columns):

    series = cut[col]
    time   = cut.index

    plt.subplot( len(cut.columns), 1, ind+1 )
    plt.plot(time, series)
    plt.ylabel( col )
    
plt.savefig("/home/yair/lasmap/huisman/pix/python/restricted_time_series.png")
plt.close()

######################################################################
## Find rho for predicting using different embedding dimensions (and,
## of course) plot.
######################################################################
Es = range(1,11)
rhos = np.empty( len(Es) )
for variable in df.columns:
    print variable
    uni = pd.DataFrame(data=df[variable],
                       columns=[variable],
                       index= df.index)

    for ind, E in enumerate(Es):
        preds = simplex.univariate(uni,E)
        rhos[ind] = preds["pred"].corr(preds["obs"])
    
    plt.plot(Es, rhos)
    plt.ylabel("Prediction skill - rho")
    plt.xlabel("Embedding dimension - E")
    plt.savefig("/home/yair/lasmap/huisman/pix/python/" + variable + "_skill_full.png")
    plt.close()




## Read entire data
df = pd.read_csv( "data/huisman.csv" )

## Restrict data, because we don't want to use ALL of it. Huisman data
## goes from 10 to 6000, in increments of 10. Hence length of data
## frame is 600.
df = df.iloc[300:400]

## Normalize
df = helpers.normalize(df)

## Add noise...
# df = helpers.add_noise(df,
#                        sig=math.sqrt(0.1),
#                        time="time")

## Lag coordinates X(t), X(t-1), X(t-2). Three lags.
df = helpers.lag(df, 3)

betas = comp.get_betas("N1",
                       df )
print( betas.iloc[4] )
