from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation
from keras.layers.merge import dot
from keras.models import Model
from keras import backend as K

from utility import globalvars

import tensorflow as tf


def create_network_3(input_shape, nb_classes, nb_lstm_cells=128):
    '''
    input_shape: (time_steps, features,)

    Dense layers -> No activation
    BLSTM -> activation softsign
    '''

    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        input_feature = Input(shape=input_shape)
        x = Masking(mask_value=globalvars.masking_value)(input_feature)
        x = LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5)(x)
        y = LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5)(x)

    with K.name_scope('AttentionLayer'):
        # Logistic regression for learning the attention parameters with a standalone feature as input
        input_attention = Input(shape=(nb_lstm_cells * 2,))
        u = Dense(nb_lstm_cells * 2, activation='softsign')(input_attention)

        # To compute the final weights for the frames which sum to unity
        alpha = dot([u, y], axes=-1)  # inner prod.
        alpha = Activation('softmax')(alpha)

    with K.name_scope('WeightedPooling'):
        # Weighted pooling to get the utterance-level representation
        z = dot([alpha, y], axes=1)

    with K.name_scope('OUTPUT'):
        # Get posterior probability for each emotional class
        output = Dense(nb_classes, activation='softmax')(z)

    model = Model(inputs=[input_attention, input_feature], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_network_2(input_shape, nb_classes, nb_lstm_cells=128):
    '''
    input_shape: (time_steps, features,)

    Dense layers -> No activation
    BLSTM -> activation softsign
    '''

    tf.logging.set_verbosity(tf.logging.ERROR)  # evitar warnings por cambio de versión

    with K.name_scope('BLSTMLayer'):
        # Bi-directional Long Short-Term Memory for learning the temporal aggregation
        input_feature = Input(shape=input_shape)
        x = Masking(mask_value=globalvars.masking_value)(input_feature)
        x = Dense(globalvars.nb_hidden_units)(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(globalvars.nb_hidden_units)(x)
        x = Dropout(rate=0.5)(x)
        y = Bidirectional(LSTM(nb_lstm_cells, return_sequences=True, dropout=0.5))(x)

    with K.name_scope('AttentionLayer'):
        # Logistic regression for learning the attention parameters with a standalone feature as input
        input_attention = Input(shape=(nb_lstm_cells * 2,))
        u = Dense(nb_lstm_cells * 2, activation='softsign')(input_attention)

        # To compute the final weights for the frames which sum to unity
        alpha = dot([u, y], axes=-1)  # inner prod.
        alpha = Activation('softmax')(alpha)

    with K.name_scope('WeightedPooling'):
        # Weighted pooling to get the utterance-level representation
        z = dot([alpha, y], axes=1)

    with K.name_scope('OUTPUT'):
        # Get posterior probability for each emotional class
        output = Dense(nb_classes, activation='softmax')(z)

    model = Model(inputs=[input_attention, input_feature], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


"""
Function from:
https://github.com/RayanWang/Speech_emotion_recognition_BLSTM/blob/master/utility/networks.py
"""


# Red original SER_BLSTM
def create_network_1(input_shape, nb_classes, nb_lstm_cells=128):
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

    model = Model(inputs=[input_attention, input_feature], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
