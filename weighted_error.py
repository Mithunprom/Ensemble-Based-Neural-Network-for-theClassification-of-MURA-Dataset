import numpy as np

"""Function for calculating weighted error Epsilon""" #approved
def weighted_error(y,ypred,d):
    t = len(d)
    vec = np.abs(np.subtract(y,ypred))
    sumerror = 0.0
    for i in range (t):
        sumerror = sumerror + (d[i]*vec[i])
    return sumerror
