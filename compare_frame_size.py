import argparse
import numpy as np
import pandas as pd
import sys
import os
from scipy import stats
from dataset import Dataset
from pyAudioAnalysis import audioFeatureExtraction

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
        f = np.mean(f, axis=0)

        f_global.append(f)

        sys.stdout.write("\033[F")
        i = i + 1
        print("\t Extracting features " + str(i) + '/' + str(nb_samples) + " from data set...")

    return f_global


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="name of dataset")
    parser.add_argument("gender", help="male female both")
    parser.add_argument("path", help="path to dataset")
    parser.add_argument('-emotions', action='store', nargs='*',
                        default=[],
                        help='anger disgust fear happiness sadness surprise')
    args = parser.parse_args()

    name_dataset = args.name
    path_dataset = args.path
    gender = args.gender
    emotions = args.emotions

    emotion_dic = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'sadness': 4, 'surprise': 5}
    emotions.sort()
    number_emo = []
    for emo in emotions:
        number_emo.append(emotion_dic[emo])
    name_save_csv = name_dataset + "-" + gender + '-' + ''.join(str(e) for e in number_emo)

    dataset = Dataset(path_dataset, name_dataset, gender, emotions, number_emo, 0.03, 0.015)
    df_emotions = pd.Series(data=dataset.targets, name="emotion", dtype="category")
    df_emotions = df_emotions.map(lambda i: dataset.dictionary[i])

    labels = [
        'zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread',
        'spectral_entropy', 'spectral_flux', 'spectral_rolloff', 'mfcc_1',
        'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8',
        'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1',
        'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7',
        'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12',
        'chroma_std', 'harmonic_ratio', 'pitch'
    ]

    df = pd.DataFrame()
    sizes = np.arange(0.015, 0.04, step=0.001)
    for i in sizes:
        print("frame_size = " + str(i))
        print("step_size = " + str(i / 2))
        dataset = Dataset(path_dataset, name_dataset, gender, emotions, number_emo, i, i / 2)
        features = extract_features(dataset)
        df_features = pd.DataFrame(data=features, columns=labels)
        df_aux = pd.concat([df_emotions, df_features], axis=1)
        df_aux["frame_size"] = pd.Series(np.full(shape=df_aux.size, fill_value=i))
        df_aux = df_aux.groupby('emotion').mean()
        df_aux["emotion"] = df_aux.index
        df = df.append(df_aux, ignore_index=True)
        print(df)

    # source: https://thispointer.com/how-to-create-a-directory-in-python/
    # Create target directory & all intermediate directories if don't exists
    directory = "data/compare_frame_size_csv/"
    try:
        os.makedirs(directory)
        print("Directory ", directory, " Created ")
    except FileExistsError:
        print("Directory ", directory, " already exists")
    df.to_csv(path_or_buf=directory + name_save_csv + ".csv", index=False)
