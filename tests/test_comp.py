import numpy as np
import pandas as pd
import pdb
import sys
sys.path.append( '~/LasMap/')

from lasmap import comp

def test_lasso_map():

    ## Test many a times we actually get the correct number of
    ## non-zero coefficients
    for _ in range(100):
        ## Design matrix (data frame)
        # X = np.array([ [1,3,4,6,5,6],
        #                [4,3,6,9,0,3],
        #                [2,4,5,3,1,8],
        #                [8,9,0,8,0,3],
        #                [2,5,6,1,2,7],
        #                [1,7,8,3,4,5] ], dtype = np.float64 )
        X = np.random.randn( 210, 12 )
        X = pd.DataFrame( X )
        # X.columns = ["a", "b", "c", "d", "e", "f"]

        ## True beta and E
        beta = np.random.randn( X.shape[1] ) * np.random.choice([0, 1], size=(X.shape[1],), p=[1./3, 2./3])
        E = np.random.randint(low=2,high=5)

        ## Noisy observations with intercept added
        y = np.einsum( "ij, j -> i", X, beta ) + 2 
        y = y + np.random.randn( len(y) ) * 2
        
        ## A random vector in R^d
        x = np.random.randn( X.shape[1] )

        beta_hat = comp.lasso_map(X,
                                  y,
                                  x,
                                  E=E,
                                  theta=np.random.uniform(0,5) )

        ## Did we get what we wanted?
        assert np.sum(beta_hat) == E
