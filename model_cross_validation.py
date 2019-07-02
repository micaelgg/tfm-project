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
    # almacenar datos de ejecución
    parser = ArgumentParser()
    parser.add_argument('name', action='store',
                        help='Name of dataset')
    parser.add_argument('network_number', action='store',
                        help='Network')
    args = parser.parse_args()
    globalvars.dataset = args.name
    network_number = int(args.network_number)
    dataset = args.name
    dataset_path = "data/" + dataset + "/"

    # crear directorio del modelo
    start_time = datetime.now().strftime("%H:%M")
    model_path = dataset_path + datetime.now().strftime("%d-%m-%y_%H:%M") + "/"
    try:
        os.makedirs(model_path)
        print("Directory ", model_path, " Created ")
    except FileExistsError:
        print("Directory ", model_path, " already exists")
    logging.basicConfig(filename=model_path + 'model_cross_validation.log',
                        level=logging.INFO,
                        format="%(message)s")

    # cargar dataset y features
    logging.info("\t\tModel cross validation\n")
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

    logging.info("\nLoading features from file...")
    f_global_all_features = pickle.load(open(dataset_path + dataset + '_features_sequence.p', 'rb'))

    # eliminar variables no deseadas
    removed_features = np.arange(21, 34, 1)
    label_features = np.delete(globalvars.label_features, removed_features)
    logging.info("features = %s" % label_features)
    shape = [f_global_all_features.shape[0],
             f_global_all_features.shape[1],
             f_global_all_features.shape[2] - len(removed_features)]
    f_global = np.zeros(shape)
    for i in range(f_global_all_features.shape[0]):
        f_global[i] = np.delete(f_global_all_features[i], removed_features, axis=1)

    # cargar y adecuar el formato de la variable de salida
    y = np.array(ds.targets)
    y = to_categorical(y)

    # crear los folds
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
    logging.info("\nUsing speaker independence %s-fold cross validation" % k_folds)

    # Entrenamiento de NN
    logging.info("globalvars.attention_init_value = %s" % globalvars.attention_init_value)
    logging.info("globalvars.nb_attention_param =  %s" % globalvars.nb_attention_param)

    cvscores = []
    i = 1
    for (train, test) in splits:
        clear_session()
        # initialize the attention parameters with all same values for training and validation

        # create network
        globalvars.max_len = f_global.shape[1]
        globalvars.nb_features = f_global.shape[2]
        logging.info("\n %s-fold:" % i)

        if network_number == 1:
            logging.info("create_network_1")  # NETWORK
            model = networks.create_network_1(input_shape=(globalvars.max_len, globalvars.nb_features),
                                              nb_classes=nb_classes)
        elif network_number == 2:
            logging.info("create_network_2")  # NETWORK
            model = networks.create_network_2(input_shape=(globalvars.max_len, globalvars.nb_features),
                                              nb_classes=nb_classes)
        elif network_number == 3:
            logging.info("create_network_3")  # NETWORK
            model = networks.create_network_3(input_shape=(globalvars.max_len, globalvars.nb_features),
                                              nb_classes=nb_classes)
        elif network_number == 4:
            logging.info("create_network_4")  # NETWORK
            model = networks.create_network_5(input_shape=(globalvars.max_len, globalvars.nb_features),
                                              nb_classes=nb_classes)
        elif network_number == 5:
            logging.info("create_network_5")  # NETWORK
            model = networks.create_network_5(input_shape=(globalvars.max_len, globalvars.nb_features),
                                              nb_classes=nb_classes)
        elif network_number == 6:
            logging.info("create_network_6")  # NETWORK
            model = networks.create_network_6(input_shape=(globalvars.max_len, globalvars.nb_features),
                                              nb_classes=nb_classes)
        elif network_number == 7:
            logging.info("create_network_7")  # NETWORK
            model = networks.create_network_7(input_shape=(globalvars.max_len, globalvars.nb_features),
                                              nb_classes=nb_classes)

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
        hist = model.fit(f_global[train], y[train],
                         epochs=200,
                         batch_size=128,
                         verbose=2,
                         callbacks=callback_list,
                         validation_data=(f_global[test], y[test]))

        # evaluate the best model in ith fold
        best_model = load_model(file_path)

        logging.info("Evaluating on test set...")
        scores = best_model.evaluate(f_global[test], y[test], batch_size=128, verbose=1)
        logging.info("The highest %s in %dth fold is %.2f%%" % (model.metrics_names[1], i, scores[1] * 100))

        cvscores.append(scores[1] * 100)

        logging.info("Getting the confusion matrix on WHOLE set...")
        predictions = best_model.predict(f_global)
        confusion_matrix = metrics_util.get_confusion_matrix_one_hot(predictions, y)
        logging.info(confusion_matrix)

        logging.info("Getting the confusion matrix on TEST set...")
        predictions = best_model.predict(f_global[test])
        confusion_matrix = metrics_util.get_confusion_matrix_one_hot(predictions, y[test])
        logging.info(confusion_matrix)

        i += 1

# mensajes de fin de ejecución
logging.info("\nAccuracy: " + "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
end_time = datetime.now().strftime("%H:%M")
logging.info("Start time: " + start_time)
logging.info("End time: " + end_time)
sys.exit()
