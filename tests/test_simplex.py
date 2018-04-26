import numpy as np
import pandas as pd
from math import exp as exp
import pytest
import os

from lasmap import simplex

## Making sure directory where we save data exists
if not os.path.exists("tests/data"):
    os.makedirs("tests/data")
    
def test_generic_first():
    
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
    
    assert simplex.generic(data, obs, x) == 1.


@pytest.fixture
def simple_data():

    '''Data with known distances between data points

    '''
    
    x = np.ones(3)
    obs = np.array([1.,2.,3.,4.])
    data = np.empty( (4,3) )

    ## So first line is distance 1 from x
    v = np.array( [2,-1,3] )
    v = v / np.linalg.norm(v)
    data[0,:] = x + v

    ## So second line is distance 1/2 from x
    v = np.array( [-1,0,3])
    v = v / np.linalg.norm(v) / 2.
    data[1,:] = x + v

    ## So third line is distance 2 from x
    v = np.array( [1,6,2])
    v = v / np.linalg.norm(v) * 2.
    data[2,:] = x + v
    
    ## So fourth line is distance 3 from x
    v = np.array( [0,3,-4])
    v = v / np.linalg.norm(v) * 3.
    data[3,:] = x + v
    
    ## We want to comare our results with the rEDM package, so we save the
    ## data:
    arr          = np.empty((5,4))
    arr[0:4,0:3] = data
    arr[0:4,  3] = obs
    arr[4  ,0:3] = x
    columns = ["V1", "V2", "V3", "target" ]
    df = pd.DataFrame(columns=columns,
                      data=arr)

    df.to_csv("tests/data/simple_data.csv",
              index=False)

    return {"data" : data, "x": x, "obs": obs }
    
def test_data_is_OK(simple_data):

    ## Unpack
    data = simple_data["data"]
    x = simple_data["x"]
    
    ## Just check we didn't screw up with the linear algebra.
    assert abs(np.linalg.norm(data[0,:]-x) - 1.0) < 1e-14
    assert abs(np.linalg.norm(data[1,:]-x) - 0.5) < 1e-14
    assert abs(np.linalg.norm(data[2,:]-x) - 2.0) < 1e-14
    assert abs(np.linalg.norm(data[3,:]-x) - 3.0) < 1e-14
    
def test_generic_two_nearest_neighbours(simple_data):

    ## Unpack
    data = simple_data["data"]
    x = simple_data["x"]
    obs = simple_data["obs"]
    
    ## Using two nearest neighbours simplex the following calculation
    ## predicts the observable at x:
    calc = (1 * exp(-1/0.5) + 2 * exp(-0.5/0.5) ) / ( exp(-1/0.5) + exp(-0.5/0.5) )

    ## Calling the function, we get:
    func = simplex.generic(data,
                           obs,
                           x,
                           num_nn=2)

    ## They need to be equal, at least up to numerical errors.
    assert abs(func - calc) < 1e-14

    fl = open("tests/data/2NN.txt","w") 
    fl.write(str(calc) + "\n" )
    
def test_generic_three_nearest_neighbours(simple_data):

    ## Unpack
    data = simple_data["data"]
    x = simple_data["x"]
    obs = simple_data["obs"]
    
    ## Similarly, using three nearest neighbours:
    calc = (1 * exp(-1/0.5) + 2 * exp(-0.5/0.5) + 3 * exp(-2./0.5) ) / ( exp(-1/0.5) + exp(-0.5/0.5) + exp(-2./0.5) )
    func = simplex.generic(data,
                           obs,
                           x,
                           num_nn=3)

    ## They better be equal...
    assert abs(func - calc) < 1e-14

    fl = open("tests/data/3NN.txt", "w") 
    fl.write(str(calc) + "\n" )

@pytest.fixture
def make_random_set_data():

    np.random.seed(19)

    ## Create a dataframe and populate it with fake data.
    lib = [1,2,5,6,8]
    pred = [9,11,13,14,16]
    columns = ["V1", "V2", "V3", "target" ]
    predictors = ["V1", "V2", "V3"]
    target = "target"
    df = pd.DataFrame(index=np.ravel([lib,pred]),
                      columns=columns,
                      data=np.random.normal(size=(len(lib)+len(pred),len(columns)))) 

    return { "lib": lib,
             "pred": pred,
             "columns": columns,
             "target": target,
             "predictors": predictors,
             "df": df }

def test_generic_sets(make_random_set_data):

    ## Unpack
    lib = make_random_set_data["lib"]
    pred = make_random_set_data["pred"]
    columns = make_random_set_data["columns"]
    target = make_random_set_data["target"]
    predictors = make_random_set_data["predictors"]
    df = make_random_set_data["df"]

    calc = simplex.generic_sets(df.reindex(labels=lib, axis="index"),## df.reindex(labels=lib, axis="index"),
                                df.reindex(labels=pred, axis="index"),
                                target,
                                predictors)

    ## Set this data in the dataframe
    df.at[pred,"target"] = calc["pred"]

    # Save...
    df.to_csv("tests/data/generic_sets.csv",
              index=False)


@pytest.fixture
def random_univariate_data():
    np.random.seed(19)

    ## Generate fake data
    n = 12
    df = pd.DataFrame(index=np.arange(n)*10,
                      columns=["A"],
                      data=np.random.randint(-2*n,2*n,n))
    return df

def test_univariate(random_univariate_data):
    
    preds = simplex.univariate(random_univariate_data,
                               2,
                               tp=2)
    preds["truth"] = random_univariate_data["A"]
    preds.to_csv("tests/data/univariate.csv",
                 index=True,
                 index_label="time",
                 na_rep = "NaN")

