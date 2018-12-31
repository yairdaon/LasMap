#!/usr/bin/python
import numpy as np
import pandas as pd
import pdb
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

import lasmap.helpers as helpers
import lasmap.simplex as simplex

if not os.path.exists("Huisman/pix"):
    os.makedirs("Huisman/pix/python")
    os.makedirs("Huisman/pix/R")
os.system("rm -f Huisman/pix/python/*")

    
## For reproducibility purposes
np.random.seed(89519241)

## Read entire data
raw = pd.read_csv("Huisman/raw_noiseless_huisman.csv",
                 index_col="time")
raw = raw.drop(["R1", "R2", "R3"], axis=1 )

raw["N2+N3"] = raw["N2"] + raw["N3"]
raw["N4+N5"] = raw["N4"] + raw["N5"]

## Normalize, ***then*** add noise (with prescribed standard
## deviation).
std = 0.1
df = helpers.add_noise(helpers.normalize(raw), std)

## In this region, N2 is approximately constant.
N2_const = df.loc[3550:3790]

## Restrict data. If we use all data, then the "true" nature of the
## system is revealed: the embedding dimension should be 8 (five
## species + 3 abiotic variables). ***Skill actually peaks at
## 5-6***. This can be seen by commenting the line below and setting std
## to zero. We imitate the case where time series is too short and so
## the embedding dimension seems smaller than it really is.
df = df.loc[1000:2500]

## These are the variables of interest. These are coupled (i.e. are
## influenced by both "lobes")
coupled = ["N2+N3", "N4+N5", "N1"]

################################################################
## Plot all time series on all of their time span.
################################################################
for ind, col in enumerate(df.columns):

    series = df[col]
    time   = df.index

    plt.subplot( len(df.columns), 1, ind+1 )
    plt.plot(time, series)
    plt.ylabel( col )
    
plt.savefig("Huisman/pix/python/time_series.png")
plt.close()

################################################################
## Plot all time series on the restricted region, where N2 is
## approximately constant
################################################################
for ind, col in enumerate(N2_const.columns):

    series = N2_const[col]
    time   = N2_const.index

    plt.subplot( len(N2_const.columns), 1, ind+1 )
    plt.plot(time, series)
    plt.ylabel( col )
    
plt.savefig("Huisman/pix/python/restricted_time_series.png")
plt.close()

######################################################################
## Find rho for predicting using different embedding dimensions (and,
## of course) plot.
######################################################################
Es = range(1,11)
rhos = np.empty( len(Es) )
for variable in coupled:
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
    plt.savefig("Huisman/pix/python/" + variable + "_skill.png")
    plt.close()


## Bottom line(s): N1 shows peak predictability ( rho ~ 0.8 ) with
## embedding dimension E = 3. For N2+N4 peak is at E = 2 (rho >
## 0.65). For N3+N5, rho peaks at E = 6.

## Let us save this data for further analysis. This dataframe has
## Huisman data that is normalized, noisy and truncated. Or, in a
## word: processed. It is ***not*** lagged and does not have an
## observation (i.e. _p1) column. It ***does*** have summed columns
## (i.e. N2+N3 and N4+N5).
df.to_csv("Huisman/processed_huisman.csv")

