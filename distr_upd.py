import numpy as np

"""Function for updating distribution"""
def distr_upd(D, epsilon, alpha, y, h_x):       #approved
    z = 2.0* math.sqrt(epsilon*(1-epsilon))
    D_out = np.zeros(len(D))
    ##function to find mismatch vector
    ylabel = np.array(y)
    hypo = np.array(h_x)
    mismatch_vec = np.where(ylabel!=hypo, -1, 1)
    for i in range (len(D)):
        D_out[i] = (D[i]/z) * math.exp( (-alpha)* mismatch_vec[i])
    D_out = D_out / np.sum(D_out)
    return D_out/np.sum(D_out)
