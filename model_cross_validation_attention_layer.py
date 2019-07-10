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
from utility import networks_attention_layer, metrics_util, globalvars

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

try:
    import cPickle as pickle
except ImportError:
    import pickle

if __name__ == '__main__':
    # almacenar datos de ejecución
    parser = ArgumentParser()
    parser.add_argument('name', action='store',
                        help='Name of dataset')
    parser.add_argument('network_name', action='store',
                        help='Network')
    parser.add_argument('mini_batch', action='store',
                        help='Network')
    args = parser.parse_args()
    globalvars.dataset = args.name
    network_name = args.network_name
    dataset = args.name
    dataset_path = "data/" + dataset + "/"
    mini_batch = int(args.mini_batch)

    # crear directorio del modelo
    start_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    model_path = dataset_path + datetime.now().strftime("%d-%m-%y_%H:%M") + "/"
    try:
        os.makedirs(model_path)
        print("Directory ", model_path, " Created ")
    except FileExistsError:
        print("Directory ", model_path, " already exists")

    # variables para la construcción "manual" del log
    logging_text = "\t\tModel cross validation - Attention layer\n"
    text_1 = ""
    text_2 = ""
    text_3 = ""
    text_4 = ""

    # cargar dataset

    ds = pickle.load(open(dataset_path + dataset + '_db.p', 'rb'))
    nb_samples = len(ds.targets)
    globalvars.nb_classes = len(np.unique(ds.targets))
    nb_classes = globalvars.nb_classes

    text_1 += "\n" + "Loading data from " + dataset
    text_1 += "\n" + "name_dataset = " + ds.name_dataset
    text_1 += "\n" + "frame_size = " + str(ds.frame_size)
    text_1 += "\n" + "step_size = " + str(ds.step)
    text_1 += "\n" + "gender = " + dataset.split("-")[1]
    text_1 += "\n" + "emotions = " + str(ds.name_emotions)

    #  cargar y eliminar variables
    f_global_all_features = pickle.load(open(dataset_path + dataset + '_features_sequence.p', 'rb'))
    removed_features = np.arange(23, 36, 1)
    label_features = np.delete(globalvars.label_features, removed_features)
    shape = [f_global_all_features.shape[0],
             f_global_all_features.shape[1],
             f_global_all_features.shape[2] - len(removed_features)]
    f_global = np.zeros(shape)
    for i in range(f_global_all_features.shape[0]):
        f_global[i] = np.delete(f_global_all_features[i], removed_features, axis=1)

    text_1 += "\n\n" + "Loading features from file..."
    text_1 += "\n" + "features " + str(label_features)

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

    # Entrenamiento de NN
    cvscores = []
    i = 1
    for (train, test) in splits:
        clear_session()
        # initialize the attention parameters with all same values for training and validation
        u_train = np.full((len(train), globalvars.nb_attention_param),
                          globalvars.attention_init_value, dtype=np.float32)
        u_test = np.full((len(test), globalvars.nb_attention_param),
                         globalvars.attention_init_value, dtype=np.float32)

        # create network
        globalvars.max_len = f_global.shape[1]
        globalvars.nb_features = f_global.shape[2]

        model = networks_attention_layer.select_network(network_name)
        text_2 = "\n\n\n\t" + model.name  # NETWORK

        file_path = model_path + 'weights_' + str(i) + '_fold' + '.h5'
        callback_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=40,
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
                histogram_freq=2,
                write_graph=True,
                write_images=True
            ),
            CSVLogger(
                filename=file_path + '-fit.log'
            )
        ]

        batch_size = mini_batch
        epochs = 300

        # fit the model
        hist = model.fit([u_train, f_global[train]], y[train],
                         epochs=epochs,
                         batch_size=batch_size,
                         verbose=2,
                         callbacks=callback_list,
                         validation_data=([u_test, f_global[test]], y[test]))

        # evaluate the best model in ith fold
        best_model = load_model(file_path)

        model_json = best_model.to_json()
        with open(model_path + "best_model" + str(i) + ".json", "w") as json_file:
            json_file.write(model_json)

        scores = best_model.evaluate([u_test, f_global[test]], y[test], batch_size=batch_size, verbose=1)
        cvscores.append(scores[1] * 100)

        u = np.full((f_global.shape[0], globalvars.nb_attention_param),
                    globalvars.attention_init_value, dtype=np.float32)
        predictions_whole_set = best_model.predict([u, f_global])
        confusion_matrix_whole_set = metrics_util.get_confusion_matrix_one_hot(predictions_whole_set, y)

        predictions_test_set = best_model.predict([u_test, f_global[test]])
        confusion_matrix_test_set = metrics_util.get_confusion_matrix_one_hot(predictions_test_set, y[test])

        class_report = (classification_report(y_pred=predictions_test_set.argmax(axis=1), y_true=y[test].argmax(axis=1),
                                              target_names=ds.name_emotions))

        text_3 += "\n\n\n\t" + str(i) + "-fold:"
        text_3 += "\n" + "Evaluating on test set... "
        text_3 += "\n" + "The highest acc is " + str(round(scores[1] * 100, 2)) + "%"
        text_3 += "\n\n" + "Getting the confusion matrix on WHOLE set..."
        text_3 += "\n" + str(confusion_matrix_whole_set)
        text_3 += "\n\n" + "Getting the confusion matrix on TEST set..."
        text_3 += "\n" + str(confusion_matrix_test_set)
        text_3 += "\n\n" + "Classification_report"
        text_3 += "\n" + str(class_report)

        i += 1

text_2 += "\n" + "globalvars.attention_init_value = " + str(globalvars.attention_init_value)
text_2 += "\n" + "globalvars.nb_attention_param = " + str(globalvars.nb_attention_param)
text_2 += "\n" + str(json.dumps(hist.params))
tmp_smry = StringIO()
model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
text_2 += "\n" + tmp_smry.getvalue()

text_2 += "\n\n" + "Using speaker independence" + str(k_folds) + "-fold cross validation"
text_2 += "\n" + str(kfold)

plot_model(model, to_file=model_path + 'model_plot.png', show_shapes=True, show_layer_names=True)

# mensajes de fin de ejecución
end_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
text_4 += "\n\n\n" + "\tUsing speaker independence " + str(k_folds) + "-fold cross validation"
text_4 += "\n" + "Accuracy = " + str(round(np.mean(cvscores), 2))
text_4 += "\n" + "Standard deviation = " + str(round(np.std(cvscores), 2))
text_4 += "\n\n" + "Start time: " + start_time
text_4 += "\n" + "End time: " + end_time

# guardar logging
logging_text += text_1 + text_2 + text_3 + text_4
f = open(model_path + 'model_cross_validation.log', 'w')
f.write(logging_text)
f.close()

sys.exit()
