import argparse
import pickle
import numpy as np
import sys
import os
from sklearn import preprocessing
from dataset import Dataset
import generate_csv
import librosa
from keras.preprocessing import sequence
import pandas as pd
from utility import globalvars

""" 
Function from:
https://github.com/RayanWang/Speech_emotion_recognition_BLSTM/blob/master/utility/audio.py
"""


def extract_features(dataset):
    data = dataset.data
    nb_samples = len(dataset.targets)
    frame_size = dataset.frame_size
    step = dataset.step
    f_global = []

    i = 0
    for (x, Fs) in data:
        # ,hop_length=hop_length

        mfcc = librosa.feature.mfcc(x, n_mfcc=64)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc = mfcc.transpose()
        mfcc_delta = mfcc_delta.transpose()
        mfcc_delta2 = mfcc_delta2.transpose()

        mfcc = preprocessing.MinMaxScaler().fit_transform(mfcc)
        mfcc_delta = preprocessing.MinMaxScaler().fit_transform(mfcc_delta)
        mfcc_delta2 = preprocessing.MinMaxScaler().fit_transform(mfcc_delta2)

        mfcc = mfcc.transpose()
        mfcc_delta = mfcc_delta.transpose()
        mfcc_delta2 = mfcc_delta2.transpose()

        mfcc = sequence.pad_sequences(mfcc,
                                      maxlen=globalvars.max_len,
                                      dtype='float32',
                                      padding='post',
                                      value=0.0)

        mfcc_delta = sequence.pad_sequences(mfcc_delta,
                                            maxlen=globalvars.max_len,
                                            dtype='float32',
                                            padding='post',
                                            value=0.0)

        mfcc_delta2 = sequence.pad_sequences(mfcc_delta2,
                                             maxlen=globalvars.max_len,
                                             dtype='float32',
                                             padding='post',
                                             value=0.0)

        mfcc = mfcc.transpose()
        mfcc_delta = mfcc_delta.transpose()
        mfcc_delta2 = mfcc_delta2.transpose()

        f_global.append([mfcc, mfcc_delta, mfcc_delta2])

        # todo normalizar con todo el datset

        sys.stdout.write("\033[F")
        i = i + 1
        print("Extracting features " + str(i) + '/' + str(nb_samples) + " from data set...")

    return np.array(f_global)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="name of dataset")
    parser.add_argument("gender", help="male female both")
    parser.add_argument("frame_size", help="frame size (milliseconds)")
    parser.add_argument("path", help="path to dataset")
    parser.add_argument('-emotions', action='store', nargs='*',
                        default=[],
                        help='anger disgust fear happiness sadness surprise')
    args = parser.parse_args()

    name_dataset = args.name
    path_dataset = args.path
    gender = args.gender
    emotions = args.emotions
    frame_size = float(args.frame_size) * 0.001
    step = float(frame_size) / 2

    emotion_dic = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'sadness': 4, 'surprise': 5}
    emotions.sort()
    number_emo = []
    for emo in emotions:
        number_emo.append(emotion_dic[emo])

    dataset = Dataset(path_dataset, name_dataset, gender, emotions, number_emo, frame_size, step)
    name_save_dataset = name_dataset + "-" + gender + '-' + str(frame_size) + '-' + ''.join(
        str(e) for e in number_emo) + "_mfcc_deltas"
    path_save_dataset = "data/" + name_save_dataset + "/" + name_save_dataset

    # source: https://thispointer.com/how-to-create-a-directory-in-python/
    # Create target directory & all intermediate directories if don't exists
    try:
        os.makedirs("data/" + name_save_dataset)
        print("Directory ", "data/" + name_save_dataset, " Created ")
    except FileExistsError:
        print("Directory ", "data/" + name_save_dataset, " already exists")

    print("Saving dataset info to " + path_save_dataset + "_db.p")
    pickle.dump(dataset, open(path_save_dataset + '_db.p', 'wb'))
    # pickle.dump(dataset, open(path_to_save + name_dataset + '_db.p', 'wb'))

    globalvars.max_len = 256
    features = extract_features(dataset)
    features = np.array(features)

    print("Saving features to file...")
    pickle.dump(features,
                open(path_save_dataset + '_features_sequence.p', 'wb'))
