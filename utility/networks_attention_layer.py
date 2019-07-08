from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation, Flatten, BatchNormalization
from keras.layers.merge import dot
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras.models import Sequential
from keras import optimizers
from utility import globalvars
import inspect

import tensorflow as tf


######################################################################################
######################################   LSTM   ######################################
######################################################################################


# Red original SER_BLSTM
def LSTM_1(input_shape, nb_classes, nb_lstm_cells=128):
    '''
     input_shape: (time_steps, features,)
    '''

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        input_feature = Input(shape=input_shape)
        x = Masking(mask_value=globalvars.masking_value)(input_feature)
        x = Dense(globalvars.nb_hidden_units, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(globalvars.nb_hidden_units, activation='relu')(x)
        x = Dropout(0.5)(x)
        y = Bidirectional(LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5))(x)

    with K.name_scope('AttentionLayer'):
        # Logistic regression for learning the attention parameters with a standalone feature as input
        input_attention = Input(shape=(nb_lstm_cells * 2,))
        u = Dense(nb_lstm_cells * 2, activation='softmax')(input_attention)

        # To compute the final weights for the frames which sum to unity
        alpha = dot([u, y], axes=-1)  # inner prod.
        alpha = Activation('softmax')(alpha)

    with K.name_scope('WeightedPooling'):
        # Weighted pooling to get the utterance-level representation
        z = dot([alpha, y], axes=1)

    # Get posterior probability for each emotional class
    output = Dense(nb_classes, activation='softmax')(z)

    model = Model(name=inspect.stack()[0][3], inputs=[input_attention, input_feature], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def LSTM_2(input_shape, nb_classes, nb_lstm_cells=128):
    '''
    input_shape: (time_steps, features,)

    Dense layers -> No activation
    BLSTM -> activation softsign
    '''

    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        input_feature = Input(shape=input_shape)
        nb_lstm_cells = 128
        x = Masking(mask_value=globalvars.masking_value)(input_feature)
        x = LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5)(x)
        x = LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5)(x)
        y = LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5)(x)

    with K.name_scope('AttentionLayer'):
        # Logistic regression for learning the attention parameters with a standalone feature as input
        input_attention = Input(shape=(nb_lstm_cells * 2,))
        u = Dense(nb_lstm_cells, activation='softsign')(input_attention)

        # To compute the final weights for the frames which sum to unity
        alpha = dot([u, y], axes=-1)  # inner prod.
        alpha = Activation('softmax')(alpha)

    with K.name_scope('WeightedPooling'):
        # Weighted pooling to get the utterance-level representation
        z = dot([alpha, y], axes=1)

    with K.name_scope('OUTPUT'):
        # Get posterior probability for each emotional class
        output = Dense(nb_classes, activation='softmax')(z)

    model = Model(name=inspect.stack()[0][3], inputs=[input_attention, input_feature], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# LSTM apiladas - Bidirectional
def LSTM_3(input_shape, nb_classes):
    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión

    model = Sequential(name=inspect.stack()[0][3])
    return model


def LSTM_4(input_shape, nb_classes):
    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión
    model = Sequential(name=inspect.stack()[0][3])
    return model


def LSTM_5(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def LSTM_6(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def LSTM_7(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


######################################################################################
######################################   LFLB   ######################################
######################################################################################
def LFLB_1(input_shape, nb_classes, nb_lstm_cells=64):
    model = Sequential(name=inspect.stack()[0][3])

    return model


def LFLB_2(input_shape, nb_classes, nb_lstm_cells=64):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def LFLB_3(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def LFLB_4(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def LFLB_5(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def LFLB_6(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def LFLB_7(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


######################################################################################
######################################   CNN   ######################################
######################################################################################

def CNN_1(input_shape, nb_classes, nb_lstm_cells=128):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def CNN_2(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def CNN_3(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def CNN_4(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def CNN_5(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def CNN_6(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


def CNN_7(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    return model


######################################################################################
######################################   CHOOSE   ####################################
######################################################################################

def select_network(network_name):
    network_type = network_name.split("_")[0]
    network_number = int(network_name.split("_")[1])
    if network_type == "LSTM":
        if network_number == 1:
            model = LSTM_1(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)
        elif network_number == 2:
            model = LSTM_2(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)
        elif network_number == 3:
            model = LSTM_3(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)
        elif network_number == 4:
            model = LSTM_4(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)
        elif network_number == 5:
            model = LSTM_5(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)
        elif network_number == 6:
            model = LSTM_6(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)
        elif network_number == 7:
            model = LSTM_7(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)

    if network_type == "CNN":
        if network_number == 1:
            model = CNN_1(input_shape=(globalvars.max_len, globalvars.nb_features),
                          nb_classes=globalvars.nb_classes)
        elif network_number == 2:
            model = CNN_2(input_shape=(globalvars.max_len, globalvars.nb_features),
                          nb_classes=globalvars.nb_classes)
        elif network_number == 3:
            model = CNN_3(input_shape=(globalvars.max_len, globalvars.nb_features),
                          nb_classes=globalvars.nb_classes)
        elif network_number == 4:
            model = CNN_4(input_shape=(globalvars.max_len, globalvars.nb_features),
                          nb_classes=globalvars.nb_classes)
        elif network_number == 5:
            model = CNN_5(input_shape=(globalvars.max_len, globalvars.nb_features),
                          nb_classes=globalvars.nb_classes)
        elif network_number == 6:
            model = CNN_6(input_shape=(globalvars.max_len, globalvars.nb_features),
                          nb_classes=globalvars.nb_classes)
        elif network_number == 7:
            model = CNN_7(input_shape=(globalvars.max_len, globalvars.nb_features),
                          nb_classes=globalvars.nb_classes)

    if network_type == "LFLB":
        if network_number == 1:
            model = LFLB_1(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)
        elif network_number == 2:
            model = LFLB_2(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)
        elif network_number == 3:
            model = LFLB_3(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)
        elif network_number == 4:
            model = LFLB_4(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)
        elif network_number == 5:
            model = LFLB_5(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)
        elif network_number == 6:
            model = LFLB_6(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)
        elif network_number == 7:
            model = LFLB_7(input_shape=(globalvars.max_len, globalvars.nb_features),
                           nb_classes=globalvars.nb_classes)

    return model
