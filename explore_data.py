from argparse import ArgumentParser

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from ggplot import *

try:
    import cPickle as pickle
except ImportError:
    import pickle


def pickle_to_datagrame(dataset):  # TODO como afecta globalvars aqui?
    lable_features = ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy',
                      'spectral_flux', 'spectral_rolloff', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6',
                      'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1', 'chroma_2',
                      'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10',
                      'chroma_11', 'chroma_12', 'chroma_std', 'harmonic_ratio', 'pitch']
    # TODO COMPROBAR 'harmonic_ratio', 'pitch'
    emotion_dic = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral',
                   7: 'calm', 8: 'boredom'}

    dataset_path = dataset + "/"
    print("Loading data from " + dataset + " data set...")
    ds = pickle.load(open(dataset_path + dataset + '_db.p', 'rb'))
    print("Number of samples: " + str(len(ds.targets)))
    print("Loading features from file...\n")
    features = pickle.load(open(dataset_path + dataset + '_features.p', 'rb'))

    df_emotions = pd.DataFrame(data=ds.targets, columns=["emotion"])
    df_emotions["emotion"] = df_emotions["emotion"].map(lambda i: emotion_dic[i])

    mean_features = []
    for single_audio in features:
        mean_features.append(np.apply_along_axis(np.mean, 0, single_audio))
    df_features = pd.DataFrame.from_records(data=mean_features, columns=lable_features)

    df_concat = pd.concat([df_features, df_emotions], axis=1)
    df_concat["dataset"] = ds.name_dataset
    return df_concat


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('emotions', action='store',
                        help='ID emotions')
    args = parser.parse_args()
    ids_emotions = args.emotions
    df_berlin = pickle_to_datagrame("berlin-" + ids_emotions)
    df_ravdess = pickle_to_datagrame("ravdess-" + ids_emotions)
    df_enterface = pickle_to_datagrame("enterface-" + ids_emotions)
    df_cremad = pickle_to_datagrame("cremad-" + ids_emotions)

    df = pd.concat([df_berlin, df_ravdess, df_enterface, df_cremad], axis=0)

    sns.set(style="ticks", palette="pastel")
    sns.boxplot(x="emotion", y="zcr", palette=["m", "g"], hue="dataset", data=df)
    sns.despine(offset=10, trim=True)

    plt.show()

    print("final")
