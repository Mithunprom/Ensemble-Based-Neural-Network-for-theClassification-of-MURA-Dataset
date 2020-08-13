from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

"""Function for creating a 5 layer NN. Creates, Compiles and Returns the Model"""   #approved
def make_nn(inlayersize, h1size, h2size, h3size, h4size, h5size, h6size, h7size, h8size, h9size, h10size, outlayersize):
    #Training data in 'xtr'
    #Labels in 'ytr'

    #input layer
    inp_layer = Input(shape=(inlayersize,))


    #hidden layer 1
    H1 = Dense(h1size, activation = 'relu')(inp_layer)

    #hidden layer 2
    H2 = Dense(h2size, activation='relu')(H1)

    #hidden layer 3
    H3 = Dense(h3size, activation = 'relu')(H2)

    #hidden layer 4
    H4 = Dense(h4size, activation = 'relu')(H3)
    
    #hidden layer 5
    H5 = Dense(h5size, activation = 'relu')(H4)

    #hidden layer 6
    H6 = Dense(h6size, activation = 'relu')(H5)

    #hidden layer 7
    H7 = Dense(h7size, activation = 'relu')(H6)
    
    #hidden layer 8
    H8 = Dense(h8size, activation = 'relu')(H7)
    
    #hidden layer 9
    H9 = Dense(h9size, activation = 'relu')(H8)

    #hidden layer 10
    H10 = Dense(h10size, activation = 'relu')(H9)

    #output layer
    out_layer = Dense(outlayersize, activation='sigmoid')(H10)



    ###creating the neural network
    shallow_nn = Model(inp_layer, out_layer)


    ###Compiling model
    shallow_nn.compile(optimizer='adadelta', loss='binary_crossentropy')

    ### Returning compiled model
    
    return shallow_nn
