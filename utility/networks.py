from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation, Flatten
from keras.layers.merge import dot
from keras.models import Model
from keras import backend as K
from keras.models import Sequential
from keras import optimizers
from utility import globalvars

import tensorflow as tf


def create_network_1(input_shape, nb_classes, nb_lstm_cells=128):
    '''
    input_shape: (time_steps, features,)

    A13

    '''

    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión

    model = Sequential()

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        model.add(Masking(mask_value=globalvars.masking_value, input_shape=input_shape))
        model.add(LSTM(256, activation='tanh', dropout=0.5, return_sequences=True))
        model.add(LSTM(256, activation='tanh', dropout=0.5))
        model.add(Dense(256, activation='softmax'))
        model.add(Dropout(rate=0.5))
    with K.name_scope('OUTPUT'):
        # Get posterior probability for each emotional class
        model.add(Dense(nb_classes, activation='softmax'))

    # TODO faltan l2(realmente no, solo en convolucionales) y batchNorm
    # TODO LSTM layers were regularized using dropout with a retention probability P = 0 . 5 and the gradients were clipped to lie in range [− 5 , 5 ] .

    # compile the model
    opt = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.99)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
