from argparse import ArgumentParser

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle


def pickle_to_dataframe(dataset):  # TODO como afecta globalvars aqui?
    label_features = ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy',
                      'spectral_flux', 'spectral_rolloff', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6',
                      'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1', 'chroma_2',
                      'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10',
                      'chroma_11', 'chroma_12', 'chroma_std', 'harmonic_ratio', 'pitch']
    # TODO COMPROBAR 'harmonic_ratio', 'pitch'

    dataset_path = "data/" + dataset + "/" + dataset
    print("Loading data from " + dataset + " data set...")
    ds = pickle.load(open(dataset_path + '_db.p', 'rb'))
    print("Number of samples: " + str(len(ds.targets)))
    print("Loading features from file...\n")
    features = pickle.load(open(dataset_path + '_features.p', 'rb'))

    df_emotions = pd.Series(data=ds.targets, name="emotion", dtype="category")
    df_emotions = df_emotions.map(lambda i: ds.dictionary[i])

    mean_features = []
    for single_audio in features:
        mean_features.append(np.apply_along_axis(np.mean, 0, single_audio))
    df_features = pd.DataFrame.from_records(data=mean_features, columns=label_features)

    df_concat = pd.concat([df_features, df_emotions], axis=1)
    df_concat["dataset"] = ds.name_dataset
    return df_concat


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('emotions', action='store',
                        help='ID emotions')
    args = parser.parse_args()
    ids_emotions = args.emotions
    df_berlin = pickle_to_dataframe("berlin-" + ids_emotions)
    df_ravdess = pickle_to_dataframe("ravdess-" + ids_emotions)
    df_enterface = pickle_to_dataframe("enterface-" + ids_emotions)
    df_cremad = pickle_to_dataframe("cremad-" + ids_emotions)

    df = pd.concat([df_berlin, df_ravdess, df_enterface, df_cremad], axis=0)

    sns.set(style="ticks", palette="pastel")
    sns.despine(offset=10, trim=True)

    label_features = ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy',
                      'spectral_flux', 'spectral_rolloff', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6',
                      'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1', 'chroma_2',
                      'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10',
                      'chroma_11', 'chroma_12', 'chroma_std', 'harmonic_ratio', 'pitch']

    for label in label_features:
        ax = sns.boxplot(x="emotion", y=label, palette=["m", "g"], hue="dataset", data=df)
        plt.show()

    print("final")
