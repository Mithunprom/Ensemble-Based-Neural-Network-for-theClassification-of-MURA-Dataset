import numpy as np

"""Function for Predicting Ensambled output"""   #approved
def ens_pred_nn(shallow_nn, x, y, alpha):
        nbsample, __ = np.shape(x)
        y_pred = np.zeros((nbsample))
        for nn, alph in zip(shallow_nn, alpha):
            y_temp, __ = predict_nn(nn, x, y)
            #added
            y_temp = np.array(y_temp)
            y_temp = np.where(y_temp==0, -1, 1)
            y_pred = y_pred + (alph * y_temp)
        
        y_pred = quantizer(y_pred)
        return y_pred  
