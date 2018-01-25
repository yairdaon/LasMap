## Test Normalize
if __name__ == "__main__":

        arr = np.array([ [1,1,3,4],
                         [2,4,3,6],
                         [3,2,4,5],
                         [4,8,9,0],
                         [5,2,5,6],
                         [6,1,7,8] ], dtype = np.float64 )
        df = pd.DataFrame( arr )
        df.columns = ["time", "x", "y", "z" ]
        normalized = normalize(df,time="time")
        assert np.all( np.abs(normalized.mean()[1:]) < 1e-14 ) 


## Test Lag
if __name__ == "__main__":
    arr = np.array([ [1,1,3,4],
                     [2,4,3,6],
                     [3,2,4,5],
                     [4,8,9,0],
                     [5,2,5,6],
                     [6,1,7,8] ], dtype = np.float64 )
    df = pd.DataFrame( arr )
    df.columns = ["time", "x", "y", "z" ]

    ## Lag using a dictionary
    lagged = lag(df,
                 { "x" : [0,2,3], "y" : [0,3], "z" : [0,1] },
                 time = "time" )
    print( lagged )

    ## Lag using an integer
    lagged = lag(df, 3, time="time")
    print( lagged )


