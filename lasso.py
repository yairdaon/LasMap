#!/usr/bin/python
import numpy as np
import pandas as pd
import pdb
import math
import matplotlib.pyplot as plt

def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

for var_ind in range(1,6):

    var = "N" + str(var_ind)
    other = "N" + str(var_ind + 1)
    if var_ind == 5:
        other = "N1"
    
    active = pd.read_csv("active_" + var + ".csv")
    df     = pd.read_csv("noisy_huisman.csv")
    df     = pd.merge(df,active,on="time")
    df     = df.iloc[400:500]
    t = df["time"]




    plt.figure(0)
    ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
    ax1.plot(t,df[var],color="r")
    ax1.set_xlim=((min(t),max(t)))
    ax1.set_title("Time series of " + var )

    ax2 = plt.subplot2grid((3,3), (1,0), rowspan=2, colspan=3)
    ax2.plot(t,df[var + "_0"],color="r")
    ax2.plot(t,df[var + "_1"],color="b")
    ax2.plot(t,df[var + "_2"],color="g")
    ax2.plot(t,df[other + "_0"],color="c")
    ax2.plot(t,df[other + "_1"],color="k")
    ax2.plot(t,df[other + "_2"],color="m")
    ax2.set_ylim((-0.1,1.1))


    # plt.suptitle("subplot2grid")
    make_ticklabels_invisible(plt.gcf())
    plt.savefig("activity_" + var + ".png")
