import numpy as np


"""Function for calculating Kappa"""
def kappa_calc(y, y_pred):
    
    if len(y) != len(y_pred):
        print('Input array sizes must match')
        return -1
    else:    
        #parameters
        total_data = len(y)
        agree_num = len(np.where(y == y_pred)[0])    #where model and specialists agree

        #specialists saying yes and no
        spec_yes = len(np.where(np.array(y) == 1)[0])
        spec_no = total_data - spec_yes

        #model saying yes
        model_yes = len(np.where(np.array(y_pred) == 1)[0])
        model_no = total_data - model_yes

        #calculating p_o
        p_o = agree_num / total_data

        #calculate p_e [raters randomly agreeing]
        rand_yes = (spec_yes/ total_data) * (model_yes/ total_data)
        rand_no = (spec_no / total_data) * (model_no / total_data)
        p_e = rand_yes + rand_no


        #kappa
        kappa = (p_o - p_e) / (1 - p_e)
        #kappa = 0
        #print('kappa value is = ', kappa)
        #print('po value is', p_o)
        #print('agree num', agree_num)
        #print('pe value is', p_e)
        #print('rand_yes ', rand_yes)
        #print('rand_no ', rand_no)
        #print('spec_yes ' , spec_yes)
        #print('spec_no', spec_no)
        #print('model_yes ', model_yes)
        #print('model_no ', model_no)
        return kappa
