import numpy as np

"""Function for quantizing predicted class values by custom AdaBoost implementation   
(so that classes take either 0 or 1 values)"""             #approved
def quantizer(y):
    yout=[]
    for i in range (len(y)):
        if(y[i]<0.0):
            yout.append(0.0)
        else:
            yout.append(1.0)
    yout = np.array(yout)
    return yout
