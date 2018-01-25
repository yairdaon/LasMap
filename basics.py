#!/usr/bin/python
import numpy as np
import pandas as pd
import pdb
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

import helpers

## Read entire data
df = pd.read_csv( "data/huisman.csv" )

## Restrict data, because we don't want to use ALL of it. Huisman data
## goes from 10 to 6000, in increments of 10. Hence length of data
## frame is 600.

## In this region, N2 is approximately constant.
df = df.iloc[355:379]

## Normalize
# df = helpers.normalize(df,time="time")

var_list =  [1,2,3,4,5]
for var_ind in var_list:
    
    series = df["N" + str(var_ind)]
    time   = df['time']

    plt.subplot( len(var_list), 1, var_ind )
    plt.plot(time, series)
    plt.ylabel( "N" + str(var_ind) )

    
plt.savefig("pix/time_series.png")
plt.show()
plt.close()


