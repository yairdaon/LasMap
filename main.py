#!/usr/bin/python
import numpy as np
import pandas as pd
import pdb
import math
import matplotlib.pyplot as plt

import helpers
import comp

## Read entire data
df = pd.read_csv( "data/huisman.csv" )

## Restrict data, because we don't want to use ALL of it. Huisman data
## goes from 10 to 6000, in increments of 10. Hence length of data
## frame is 600.
df = df.iloc[300:400]

## Normalize
df = helpers.normalize(df,time="time")

## Add noise...
# df = helpers.add_noise(df,
#                        sig=math.sqrt(0.1),
#                        time="time")

## Lag coordinates X(t), X(t-1), X(t-2). Three lags.
df = helpers.lag(df, 3, time="time")

betas = comp.get_betas("N1",
                       df )
print( betas.iloc[4] )
