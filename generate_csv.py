from argparse import ArgumentParser
import pandas as pd
import numpy as np
import os

try:
    import cPickle as pickle
except ImportError:
    import pickle

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name', action='store',
                        help='Name of dataset')
    args = parser.parse_args()
    dataset = args.name
    dataset_path = "data/" + dataset + "/"

    print("Loading data from " + dataset + " data set...")
    ds = pickle.load(open(dataset_path + dataset + '_db.p', 'rb'))
    number_instances = len(ds.targets)

    print("Loading features from file...\n")
    features = pickle.load(open(dataset_path + dataset + '_features.p', 'rb'))

    # features.csv
    label_features = [
        'zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread',
        'spectral_entropy', 'spectral_flux', 'spectral_rolloff', 'mfcc_1',
        'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8',
        'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1',
        'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7',
        'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12',
        'chroma_std', 'harmonic_ratio', 'pitch'
    ]

    df = pd.DataFrame()
    audio_index = []
    row_index = []
    i = 0
    for single_audio in features:
        audio_index = np.full(shape=len(single_audio), fill_value=i)
        row_index = np.arange(len(single_audio))
        index_df = pd.DataFrame({"audio_number": audio_index, "frame": row_index})

        single_audio_df = pd.DataFrame(single_audio, columns=label_features)

        aux_df = pd.concat([index_df, single_audio_df], axis=1)

        df = pd.concat([df, aux_df], axis=0, ignore_index=True)
        i += 1
    df = df.set_index(["audio_number", "frame"])

    directory = "data/csv/"
    name_save_csv = "features-" + dataset
    df.to_csv(path_or_buf=dataset_path + name_save_csv + ".csv", index=True)

    # emotions.csv
    df_emotions = pd.DataFrame(data=ds.targets, columns=["emotion"], dtype="category")
    df_emotions["emotion"] = df_emotions["emotion"].map(lambda i: ds.dictionary[i])
    df_emotions["audio_number"] = df_emotions.index
    df_emotions["subject"] = 99

    i = 0
    for j in ds.subjects:
        df_emotions.loc[j, "subject"] = i
        i += 1
    name_save_csv = "emotions-" + dataset
    df_emotions.to_csv(path_or_buf=dataset_path + name_save_csv + ".csv", index=False)

    # subjects_sex.csv
    df_subjects_sex = pd.DataFrame(data=ds.subjects_sex, columns=["sex"], dtype="category")
    df_subjects_sex["subject"] = df_subjects_sex.index
    name_save_csv = "subjects_sex-" + dataset
    df_subjects_sex.to_csv(path_or_buf=dataset_path + name_save_csv + ".csv", index=False)
