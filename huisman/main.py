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
df = pd.read_csv("~/lasmap/huisman/noisy_truncated_huisman.csv",
                 index_col="time")

betas = comp.get_betas("N1",
                       df )
print( betas.iloc[4] )
