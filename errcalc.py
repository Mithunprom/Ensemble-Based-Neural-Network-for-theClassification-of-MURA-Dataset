import numpy as np


"""Function for calculating error"""         #approved
def errcalc(y,ypred):
    if len(y) != len(ypred):
        print('The inputs must be same size')
        return -1
    else:
        error = np.sum(np.absolute(np.subtract(y,ypred)))
    return error/(len(y))
