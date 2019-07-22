import argparse
import pickle
import numpy as np
import sys
import os
from sklearn import preprocessing
from dataset import Dataset
import generate_csv
from pyAudioAnalysis import audioFeatureExtraction
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
    f_global_concatenate = pd.DataFrame()

    i = 0
    for (x, Fs) in data:
        # 34D short-term feature
        f = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size * Fs, step * Fs)

        # for pyAudioAnalysis which support python3
        if type(f) is tuple:
            f = f[0]

        # Harmonic ratio and pitch, 2D
        hr_pitch = audioFeatureExtraction.stFeatureSpeed(x, Fs, frame_size * Fs, step * Fs)
        f = np.append(hr_pitch.transpose(), f, axis=0)

        f = f.transpose()

        f_global.append(f)

        f_global_concatenate = pd.concat([f_global_concatenate, pd.DataFrame(f)], axis=0, ignore_index=True)

        sys.stdout.write("\033[F")
        i = i + 1
        print("Extracting features " + str(i) + '/' + str(nb_samples) + " from data set...")

    return f_global, f_global_concatenate


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

    emotion_dic = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6,
                   'calm': 7, 'boredom': 8}
    emotions.sort()
    number_emo = []
    for emo in emotions:
        number_emo.append(emotion_dic[emo])

    dataset = Dataset(path_dataset, name_dataset, gender, emotions, number_emo, frame_size, step)
    name_save_dataset = name_dataset + "-" + gender + '-' + str(frame_size) + '-' + ''.join(
        str(e) for e in number_emo) + "-all-all"
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

    features, features_concatenate = extract_features(dataset)

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(features_concatenate)

    for i in range(len(features)):
        features[i] = scaler.transform(features[i])

    print("Saving features to file...")
    pickle.dump(features, open(path_save_dataset + '_features.p', 'wb'))

    features = sequence.pad_sequences(features,
                                      maxlen=globalvars.max_len,
                                      dtype='float32',
                                      padding='post',
                                      value=globalvars.masking_value)
    print("Saving features to file... [sequence]")
    pickle.dump(features, open(path_save_dataset + '_features_sequence.p', 'wb'))

    # generate_csv.generate_csv(name_save_dataset)
