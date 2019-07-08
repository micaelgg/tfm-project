from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation, Flatten, BatchNormalization, \
    Reshape, TimeDistributed
from keras.layers.merge import dot
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, ZeroPadding2D, ConvLSTM2D
from keras import backend as K
from keras.models import Sequential
from keras import optimizers
from utility import globalvars
import inspect

import tensorflow as tf


######################################################################################
######################################   LSTM   ######################################
######################################################################################


# Red original SER_BLSTM sin capa de atencion
def LSTM_1(input_shape, nb_classes, nb_lstm_cells=128):
    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión

    model = Sequential(name=inspect.stack()[0][3])

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        model.add(Masking(mask_value=globalvars.masking_value, input_shape=input_shape))
        model.add(Dense(globalvars.nb_hidden_units, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(globalvars.nb_hidden_units, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Bidirectional(LSTM(nb_lstm_cells, dropout=0.5)))

    with K.name_scope('OUTPUT'):
        # Get posterior probability for each emotional class
        model.add(Dense(nb_classes, activation='softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model


# LSTM apiladas
def LSTM_2(input_shape, nb_classes):
    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión
    model = Sequential(name=inspect.stack()[0][3])

    nb_lstm_cells = 64

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        model.add(Masking(mask_value=globalvars.masking_value, input_shape=input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(LSTM(nb_lstm_cells, activation='softsign', dropout=0.25, return_sequences=True))
        model.add(LSTM(nb_lstm_cells, activation='softsign', dropout=0.25, return_sequences=True))
        model.add(LSTM(nb_lstm_cells, activation='softsign', dropout=0.25))

    with K.name_scope('OUTPUT'):
        # Get posterior probability for each emotional class
        model.add(Dense(nb_classes, activation='softmax'))

    # compile the model
    opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    return model


# LSTM apiladas con BatchNormalization antes de activacion
def LSTM_3(input_shape, nb_classes):
    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión
    model = Sequential(name=inspect.stack()[0][3])

    nb_lstm_cells = 64

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        model.add(Masking(mask_value=globalvars.masking_value, input_shape=input_shape))
        model.add(Dense(128, activation='relu'))

        model.add(LSTM(nb_lstm_cells, dropout=0.75, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Activation('softsign'))

        model.add(LSTM(nb_lstm_cells, dropout=0.75, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Activation('softsign'))

        model.add(LSTM(nb_lstm_cells, dropout=0.75))
        model.add(BatchNormalization())
        model.add(Activation('softsign'))

    with K.name_scope('OUTPUT'):
        # Get posterior probability for each emotional class
        model.add(Dense(nb_classes, activation='softmax'))

    # compile the model
    opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    return model


# LSTM apiladas con BatchNormalization despues de activacion
def LSTM_4(input_shape, nb_classes):
    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión
    model = Sequential(name=inspect.stack()[0][3])

    nb_lstm_cells = 64

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        model.add(Masking(mask_value=globalvars.masking_value, input_shape=input_shape))
        model.add(Dense(128, activation='relu'))

        model.add(LSTM(nb_lstm_cells, dropout=0.6, return_sequences=True))
        model.add(Activation('softsign'))
        model.add(BatchNormalization())

        model.add(LSTM(nb_lstm_cells, dropout=0.6, return_sequences=True))
        model.add(Activation('softsign'))
        model.add(BatchNormalization())

        model.add(LSTM(nb_lstm_cells, dropout=0.6))
        model.add(Activation('softsign'))
        model.add(BatchNormalization())

    with K.name_scope('OUTPUT'):
        # Get posterior probability for each emotional class
        model.add(Dense(nb_classes, activation='softmax'))

    # compile the model
    opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    return model


def LSTM_5(input_shape, nb_classes):
    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión
    model = Sequential(name=inspect.stack()[0][3])

    nb_lstm_cells = 128

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        model.add(Masking(mask_value=globalvars.masking_value, input_shape=input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(LSTM(nb_lstm_cells, activation='softsign', dropout=0.25, return_sequences=True))
        model.add(LSTM(nb_lstm_cells, activation='softsign', dropout=0.25, return_sequences=True))
        model.add(LSTM(nb_lstm_cells, activation='softsign', dropout=0.25))

    with K.name_scope('OUTPUT'):
        # Get posterior probability for each emotional class
        model.add(Dense(nb_classes, activation='softmax'))

    # compile the model
    opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    return model


def LSTM_6(input_shape, nb_classes):
    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión
    model = Sequential(name=inspect.stack()[0][3])

    nb_lstm_cells = 32

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        model.add(Masking(mask_value=globalvars.masking_value, input_shape=input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(LSTM(nb_lstm_cells, activation='softsign', dropout=0.25, return_sequences=True))
        model.add(LSTM(nb_lstm_cells, activation='softsign', dropout=0.25, return_sequences=True))
        model.add(LSTM(nb_lstm_cells, activation='softsign', dropout=0.25))

    with K.name_scope('OUTPUT'):
        # Get posterior probability for each emotional class
        model.add(Dense(nb_classes, activation='softmax'))

    # compile the model
    opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    return model


def LSTM_7(input_shape, nb_classes):
    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión
    model = Sequential(name=inspect.stack()[0][3])

    nb_lstm_cells = 128

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        model.add(Masking(mask_value=globalvars.masking_value, input_shape=input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(
            LSTM(nb_lstm_cells, activation='softsign', dropout=0.25, recurrent_dropout=0.25, return_sequences=True))
        model.add(
            LSTM(nb_lstm_cells, activation='softsign', dropout=0.25, recurrent_dropout=0.25, return_sequences=True))
        model.add(LSTM(nb_lstm_cells, activation='softsign', dropout=0.25, recurrent_dropout=0.25))

        with K.name_scope('OUTPUT'):
            # Get posterior probability for each emotional class
            model.add(Dense(nb_classes, activation='softmax'))

        # compile the model
        opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
        return model


######################################################################################
######################################   LFLB   ######################################
######################################################################################
def LFLB_1(input_shape, nb_classes, nb_lstm_cells=64):
    model = Sequential(name=inspect.stack()[0][3])

    # LFLB1
    model.add(Conv1D(filters=64, kernel_size=(3), strides=1, padding='same', data_format='channels_last',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=4, strides=4))

    # LFLB2
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=4, strides=4))

    # LFLB3
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=4, strides=4))

    # LFLB4
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=4, strides=4))

    # LSTM
    model.add(LSTM(units=256))

    # FC
    model.add(Dense(units=nb_classes, activation='softmax'))

    # Model compilation
    opt = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def LFLB_2(input_shape, nb_classes, nb_lstm_cells=64):
    model = Sequential(name=inspect.stack()[0][3])
    input_shape = input_shape
    # LFLB1
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # LFLB2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # LFLB3
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # LSTM
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(units=256))

    # FC
    model.add(Dense(units=nb_classes, activation='softmax'))

    # Model compilation
    opt = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def LFLB_3(input_shape, nb_classes):
    model = Sequential(name=inspect.stack()[0][3])
    input_shape = input_shape
    # LFLB1
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # LFLB2
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # LFLB3
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # LSTM
    # model.add(LSTM(units=256))

    # FC
    model.add(Flatten())
    model.add(Dense(units=nb_classes, activation='softmax'))

    # Model compilation
    opt = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
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

    model.add(Conv1D(128, 5, padding='valid',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='valid', ))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    opt = optimizers.rmsprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
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

def select_network(network_name, input_shape):
    network_type = network_name.split("_")[0]
    network_number = int(network_name.split("_")[1])
    if network_type == "LSTM":
        if network_number == 1:
            model = LSTM_1(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)
        elif network_number == 2:
            model = LSTM_2(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)
        elif network_number == 3:
            model = LSTM_3(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)
        elif network_number == 4:
            model = LSTM_4(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)
        elif network_number == 5:
            model = LSTM_5(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)
        elif network_number == 6:
            model = LSTM_6(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)
        elif network_number == 7:
            model = LSTM_7(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)

    if network_type == "CNN":
        if network_number == 1:
            model = CNN_1(input_shape=input_shape,
                          nb_classes=globalvars.nb_classes)
        elif network_number == 2:
            model = CNN_2(input_shape=input_shape,
                          nb_classes=globalvars.nb_classes)
        elif network_number == 3:
            model = CNN_3(input_shape=input_shape,
                          nb_classes=globalvars.nb_classes)
        elif network_number == 4:
            model = CNN_4(input_shape=input_shape,
                          nb_classes=globalvars.nb_classes)
        elif network_number == 5:
            model = CNN_5(input_shape=input_shape,
                          nb_classes=globalvars.nb_classes)
        elif network_number == 6:
            model = CNN_6(input_shape=input_shape,
                          nb_classes=globalvars.nb_classes)
        elif network_number == 7:
            model = CNN_7(input_shape=input_shape,
                          nb_classes=globalvars.nb_classes)

    if network_type == "LFLB":
        if network_number == 1:
            model = LFLB_1(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)
        elif network_number == 2:
            model = LFLB_2(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)
        elif network_number == 3:
            model = LFLB_3(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)
        elif network_number == 4:
            model = LFLB_4(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)
        elif network_number == 5:
            model = LFLB_5(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)
        elif network_number == 6:
            model = LFLB_6(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)
        elif network_number == 7:
            model = LFLB_7(input_shape=input_shape,
                           nb_classes=globalvars.nb_classes)

    return model
