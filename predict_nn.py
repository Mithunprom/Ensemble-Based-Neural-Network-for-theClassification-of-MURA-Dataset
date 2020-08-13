import numpy as np

"""Function for Predicting, and calculating error"""   #approved
def predict_nn(shallow_nn, x, y):
    pred = shallow_nn.predict(x)
    y_pred = np.round(pred)
    y_pred = np.ravel(y_pred)
    valid_err = errcalc(y,y_pred)
    return y_pred, valid_err
