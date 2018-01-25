#!/usr/bin/python
import numpy as np
import simplex

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

assert simplex.block(data, obs, x) == 1.
