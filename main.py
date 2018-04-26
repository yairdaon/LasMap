#!/usr/bin/python
import numpy as np
import pandas as pd
import pdb
import math
import matplotlib.pyplot as plt

import lasmap.helpers as helpers
import lasmap.comp as comp

## Read data
df = pd.read_csv("Huisman/processed_huisman.csv",
                 index_col = "time" )

## For testing purposes only
## df = df.iloc[1:20]

## Drop these two observables because we try to find active variables
## in predicting N1 using (lags of) N1 to N5.
df = df.drop(["N2+N3", "N4+N5"], axis=1 )

## Lag coordinates X(t), X(t-1), X(t-2). Three lags, because the
## previous calculations show E should be taken to equal 3.
E = 1
df = helpers.lag(df, E)

pdb.set_trace()
betas = comp.get_betas("N1", df)

pdb.set_trace()
groups = pd.DataFrame(index = betas.index,
                      columns = df.columns )

pdb.set_trace()
for col in groups.columns:
    pdb.set_trace()
    groups[col] = betas[col]
    for lag in range(1,E): 
        groups[col] = np.logical_or( groups[col], betas[col + "_" + str(lag)] ) 

# merged = pd.DataFrame(index = betas.index, columns = ["N1", "N2N3", "N4N5"])
# merged["N1"] = groups["N1"]
# merged["N2N3"] = np.logical_or( groups["N2"], groups["N3"] )
# merged["N4N5"] = np.logical_or( groups["N4"], groups["N5"] )

df_23 = df.loc[np.logical_or( groups["N2"], groups["N3"] )]
df_45 = df.loc[np.logical_or( groups["N4"], groups["N5"] )]


## Plot time series of N1 with colors...
plt.subplot( len(df.columns), 1, 1 )
plt.plot(df_23.index, df_23["N1"], 'bo')
plt.plot(df_45.index, df_45["N1"], 'r+')
plt.plot(df.index, df["N1"], 'b')
plt.ylabel( "N1" )


## Plot the rest of the time series.
for ind, col in enumerate( ["N2", "N3", "N4", "N5"] ):

    series = df[col]
    time   = df.index

    plt.subplot( len(df.columns), 1, ind+2 )
    plt.plot(time, series)
    plt.ylabel( col )
    
plt.savefig("Huisman/pix/python/separation.png")
plt.close()
