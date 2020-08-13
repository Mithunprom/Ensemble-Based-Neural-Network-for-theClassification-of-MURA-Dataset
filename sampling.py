import numpy as np

"""Function for Sampling
Takes a 1-by-N matrix and outputs result as a one-dimensional matrix"""   #approved

def sampling(distr, n_samples):
    
    indvec = np.array(range(n_samples))
    out_ind=[]
    
    for i in range (n_samples):
        num = np.ceil(n_samples * distr[i])
        num = int(num)
        for j in range(num):
            out_ind.append(indvec[i])
    np.random.shuffle(out_ind)
    return out_ind[0:n_samples]
