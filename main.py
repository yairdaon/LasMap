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

cols = df.columns

## Lag coordinates X(t), X(t-1), ..., X(t-E+1). Three lags (or
## not...), because the previous calculations show E should be taken
## to equal 3.
E = 1
df = helpers.lag(df, E)

## Get betas for one time step fwd in time.
betas = comp.get_betas("N1", df,tp=1)

groups = pd.DataFrame(index = betas.index,
                      columns = cols )

for col in cols:
    groups[col] = betas[col + "_0"]
    for lag in range(1,E): 
        groups[col] = np.logical_or( groups[col], betas[col + "_" + str(lag)] ) 


groups["N23"] = np.logical_or( groups["N2"], groups["N3"] )
groups["N45"] = np.logical_or( groups["N4"], groups["N5"] )

df_23 = df.loc[ groups["N23"] ] 
df_45 = df.loc[ groups["N45"] ] 

## Plot time series of N1 with colors...
f, axarr = plt.subplots( len(df.columns), sharex=True )
f.suptitle( "Husiman-Weissing: Active Variables for Predicting N1 and Time Series" )
for ax in axarr:
    ax.tick_params(
        axis='y',        # changes apply to the y-axis
        which='both',    # both major and minor ticks are affected
        left=False,      # ticks along the left edge are off
        right=False,     # ticks along the right edge are off
        labelleft=False) # labels along the left edge are off

axarr[0].plot(df_23.index, df_23["N1_0"], 'bo')
axarr[0].plot(df_45.index, df_45["N1_0"], 'ro')
axarr[0].plot(df.index, df["N1_0"], 'k')
axarr[0].set(ylabel='N1')

## The N2 N3 group is colored in blue.
axarr[1].plot(df.index, df["N2_0"], 'b')
axarr[1].set(ylabel="N2")
axarr[2].plot(df.index, df["N3_0"], 'b')
axarr[2].set(ylabel="N3")

## The N2 N3 group is colored in blue.
axarr[3].plot(df.index, df["N4_0"], 'r')
axarr[3].set(ylabel="N4")
axarr[4].plot(df.index, df["N5_0"], 'r')
axarr[4].set(ylabel="N5")

f.subplots_adjust(hspace=0)

plt.savefig("Huisman/pix/python/separation.png")
plt.close()
