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
from keras.models import load_model
from utility import networks, metrics_util, globalvars

from keras.backend import clear_session

from sklearn.model_selection import KFold

from datetime import datetime

import numpy as np
import os
import sys
import logging

try:
    import cPickle as pickle
except ImportError:
    import pickle

if __name__ == '__main__':
    start_time = datetime.now().strftime("%H:%M")

    parser = ArgumentParser()
    parser.add_argument('name', action='store',
                        help='Name of dataset')
    args = parser.parse_args()
    globalvars.dataset = args.name
    dataset = args.name
    dataset_path = "data/" + dataset + "/"

    model_path = dataset_path + datetime.now().strftime("%H:%M_%m-%d-%y") + "/"
    try:
        os.makedirs(model_path)
        print("Directory ", model_path, " Created ")
    except FileExistsError:
        print("Directory ", model_path, " already exists")
    logging.basicConfig(filename=model_path + 'model_cross_validation.log', level=logging.DEBUG)

    logging.info("Model cross validation")
    logging.info("Loading data from " + dataset + " data set...")
    ds = pickle.load(open(dataset_path + dataset + '_db.p', 'rb'))
    nb_samples = len(ds.targets)
    logging.info("Number of samples: " + str(nb_samples))
    globalvars.nb_classes = len(np.unique(ds.targets))
    nb_classes = globalvars.nb_classes
    logging.info("Number of classes: " + str(globalvars.nb_classes))
    i = 0
    for name_emo in ds.name_emotions:
        logging.info(str(i) + " -> " + name_emo)
        i += 1

    logging.info("Loading features from file...")
    f_global = pickle.load(open(dataset_path + dataset + '_features_sequence.p', 'rb'))

    y = np.array(ds.targets)
    y = to_categorical(y)

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=False, random_state=1)
    train_sets = []
    test_sets = []
    for kfold_train, kfold_test in kfold.split(ds.subjects):
        k_train_sets = []
        for k in range(0, kfold_train.size):
            k_train_sets = np.concatenate((k_train_sets, ds.subjects[kfold_train[k]]), axis=None)
        train_sets.append(k_train_sets.astype(int))

        k_test_sets = []
        for k in range(0, kfold_test.size):
            k_test_sets = np.concatenate((k_test_sets, ds.subjects[kfold_test[k]]), axis=None)
        test_sets.append(k_test_sets.astype(int))

    splits = zip(train_sets, test_sets)
    logging.info("Using speaker independence %s-fold cross validation" % k_folds)

    cvscores = []

    i = 1
    for (train, test) in splits:
        # initialize the attention parameters with all same values for training and validation
        u_train = np.full((len(train), globalvars.nb_attention_param),
                          globalvars.attention_init_value, dtype=np.float32)
        u_test = np.full((len(test), globalvars.nb_attention_param),
                         globalvars.attention_init_value, dtype=np.float32)

        # create network
        globalvars.max_len = f_global.shape[1]
        globalvars.nb_features = f_global.shape[2]
        model = networks.create_softmax_la_network(input_shape=(globalvars.max_len, globalvars.nb_features),
                                                   nb_classes=nb_classes)

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        file_path = model_path + 'weights_' + str(i) + '_fold' + '.h5'
        callback_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                verbose=1,
                mode='auto'
            ),
            ModelCheckpoint(
                filepath=file_path,
                monitor='val_acc',
                save_best_only='True',
                verbose=1,
                mode='max'
            ),
            TensorBoard(
                log_dir=model_path + '/Graph/' + str(i) + "_fold",
                histogram_freq=10,
                write_graph=True,
                write_images=True
            ),
            CSVLogger(
                filename=file_path + '-fit.log'
            )
        ]

        # fit the model
        hist = model.fit([u_train, f_global[train]], y[train],
                         epochs=200,
                         batch_size=128,
                         verbose=2,
                         callbacks=callback_list,
                         validation_data=([u_test, f_global[test]], y[test]))

        # evaluate the best model in ith fold
        best_model = load_model(file_path)

        logging.info("Evaluating on test set...")
        scores = best_model.evaluate([u_test, f_global[test]], y[test], batch_size=128, verbose=1)
        logging.info("The highest %s in %dth fold is %.2f%%" % (model.metrics_names[1], i, scores[1] * 100))

        cvscores.append(scores[1] * 100)

        logging.info("Getting the confusion matrix on whole set...")
        u = np.full((f_global.shape[0], globalvars.nb_attention_param),
                    globalvars.attention_init_value, dtype=np.float32)
        predictions = best_model.predict([u, f_global])
        confusion_matrix = metrics_util.get_confusion_matrix_one_hot(predictions, y)
        logging.info(confusion_matrix)

        clear_session()
        i += 1

logging.info("Accuracy: " + "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
end_time = datetime.now().strftime("%H:%M")
logging.info("Start time: " + start_time)
logging.info("End time: " + end_time)
sys.exit()
