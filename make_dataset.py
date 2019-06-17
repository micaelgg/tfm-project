import argparse
import pickle
import numpy as np
import sys
import os
from scipy import stats
from dataset import Dataset
from pyAudioAnalysis import audioFeatureExtraction
from keras.preprocessing import sequence

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
        # 34D short-term feature
        f = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size * Fs, step * Fs)

        # for pyAudioAnalysis which support python3
        if type(f) is tuple:
            f = f[0]

        # Harmonic ratio and pitch, 2D
        hr_pitch = audioFeatureExtraction.stFeatureSpeed(x, Fs, frame_size * Fs, step * Fs)
        f = np.append(f, hr_pitch.transpose(), axis=0)

        # Z-normalized
        f = stats.zscore(f, axis=0)

        f = f.transpose()

        f_global.append(f)

        sys.stdout.write("\033[F")
        i = i + 1
        print("Extracting features " + str(i) + '/' + str(nb_samples) + " from data set...")

    return f_global


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="name of dataset")
    parser.add_argument("frame_size", help="frame size (milliseconds)")
    parser.add_argument("path", help="path to dataset")
    parser.add_argument('-emotions', action='store', nargs='*',
                        default=[],
                        help='anger disgust fear happiness sadness surprise')
    args = parser.parse_args()

    name_dataset = args.name
    path_dataset = args.path
    emotions = args.emotions
    frame_size = float(args.frame_size) * 0.001
    step = float(frame_size) / 2

    emotion_dic = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'sadness': 4, 'surprise': 5}
    emotions.sort()
    number_emo = []
    for emo in emotions:
        number_emo.append(emotion_dic[emo])

    dataset = Dataset(path_dataset, name_dataset, emotions, number_emo, frame_size, step)
    name_save_dataset = name_dataset + '-' + str(frame_size) + '-' + ''.join(str(e) for e in number_emo)
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

    features = extract_features(dataset)

    print("Saving features to file...")
    pickle.dump(features, open(path_save_dataset + '_features.p', 'wb'))

    features = sequence.pad_sequences(features,
                                      maxlen=globalvars.max_len,
                                      dtype='float32',
                                      padding='post',
                                      value=globalvars.masking_value)
    print("Saving features to file... [sequence]")
    pickle.dump(features, open(path_save_dataset + '_features_sequence.p', 'wb'))
