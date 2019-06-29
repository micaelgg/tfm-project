from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation, Flatten
from keras.layers.merge import dot
from keras.models import Model
from keras import backend as K

from utility import globalvars

import tensorflow as tf


def create_network_1(input_shape, nb_classes, nb_lstm_cells=128):
    '''
    input_shape: (time_steps, features,)
    '''

    # tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        input_feature = Input(shape=input_shape)
        x = Masking(mask_value=globalvars.masking_value)(input_feature)
        x = Dense(globalvars.nb_hidden_units)(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(globalvars.nb_hidden_units)(x)
        y = Dropout(rate=0.5)(x)
        y = Flatten()(y)

    with K.name_scope('OUTPUT'):
        # Get posterior probability for each emotional class
        output = Dense(nb_classes, activation='softmax')(y)

    return Model(inputs=input_feature, outputs=output)
