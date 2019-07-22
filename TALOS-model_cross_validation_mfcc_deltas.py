"""
Base code:
https://github.com/RayanWang/Speech_emotion_recognition_BLSTM/blob/master/model_cross_validation.py
"""

from argparse import ArgumentParser

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from keras.models import load_model, Model
from utility import networks_mfcc_deltas, metrics_util, globalvars
from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation, Flatten, BatchNormalization, \
    Reshape, TimeDistributed
from keras.layers.merge import dot
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, ZeroPadding2D, ConvLSTM2D
from keras import backend as K
from keras.models import Sequential
from keras import optimizers
from utility import globalvars
from keras.backend import clear_session
from keras.utils.vis_utils import plot_model

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from datetime import datetime

import numpy as np
import os
import sys
import json
from io import StringIO

import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

try:
    import cPickle as pickle
except ImportError:
    import pickle


def data():
    dataset = "ravdess-male-0.025-01234_mfcc_deltas"
    dataset_path = "data/" + dataset + "/"

    ds = pickle.load(open(dataset_path + dataset + '_db.p', 'rb'))
    globalvars.nb_classes = len(np.unique(ds.targets))

    # cargar y adecuar el formato de la variable de salida
    y = np.array(ds.targets)
    y = to_categorical(y)

    f_global = pickle.load(open(dataset_path + dataset + '_features_sequence.p', 'rb'))

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=False, random_state=1)
    train_sets = []
    test_sets = []
    for kfold_train, kfold_test in kfold.split(ds.subjects_audios):
        k_train_sets = []
        for k in range(0, kfold_train.size):
            k_train_sets = np.concatenate((k_train_sets, ds.subjects_audios[kfold_train[k]]), axis=None)
        train_sets.append(k_train_sets.astype(int))

        k_test_sets = []
        for k in range(0, kfold_test.size):
            k_test_sets = np.concatenate((k_test_sets, ds.subjects_audios[kfold_test[k]]), axis=None)
        test_sets.append(k_test_sets.astype(int))

    splits = zip(train_sets, test_sets)
    for (train, test) in splits:
        x_train = f_global[train]
        x_test = y[train]
        y_train = f_global[test]
        y_test = y[test]
        return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    input_shape = (3, 256, 128)
    nb_classes = 5
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
    model.add(LSTM(units={{choice([64, 128])}}))

    # FC
    model.add(Dense(units=nb_classes, activation='softmax'))

    # Model compilation
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    result = model.fit(x_train, y_train,
                       batch_size={{choice([64, 128])}},
                       epochs=2,
                       verbose=2,
                       validation_split=0.2)
    X_train, Y_train, X_test, Y_test = data()
    # get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
