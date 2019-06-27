from argparse import ArgumentParser
import pandas as pd
import numpy as np
from utility import globalvars

try:
    import cPickle as pickle
except ImportError:
    import pickle


def generate_csv(name_dataset):
    dataset_path = "data/" + name_dataset + "/"

    print("Loading data from " + name_dataset + " data set...")
    ds = pickle.load(open(dataset_path + name_dataset + '_db.p', 'rb'))
    number_instances = len(ds.targets)

    print("Loading features from file...\n")
    features = pickle.load(open(dataset_path + name_dataset + '_features.p', 'rb'))

    # features.csv
    label_features = globalvars.label_features

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
    name_save_csv = "features-" + name_dataset
    df.to_csv(path_or_buf=dataset_path + name_save_csv + ".csv", index=True)

    # emotions.csv
    df_emotions = pd.DataFrame(data=ds.targets, columns=["emotion"], dtype="category")
    df_emotions["emotion"] = df_emotions["emotion"].map(lambda i: ds.dictionary[i])
    df_emotions["audio_number"] = df_emotions.index
    df_emotions["subject"] = 99

    i = 0
    for j in ds.subjects_audios:
        df_emotions.loc[j, "subject"] = i
        i += 1
    name_save_csv = "emotions-" + name_dataset
    df_emotions.to_csv(path_or_buf=dataset_path + name_save_csv + ".csv", index=False)

    # subjects_gender.csv
    df_subjects_gender = pd.DataFrame(data=ds.subjects_gender, columns=["gender"], dtype="category")
    df_subjects_gender["subject"] = df_subjects_gender.index
    name_save_csv = "subjects_gender-" + name_dataset
    df_subjects_gender.to_csv(path_or_buf=dataset_path + name_save_csv + ".csv", index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name', action='store',
                        help='Name of dataset')
    args = parser.parse_args()
    name_dataset = args.name

    generate_csv(name_dataset)
