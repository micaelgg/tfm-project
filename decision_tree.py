from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

try:
    import cPickle as pickle
except ImportError:
    import pickle


# TODO Misma funcion que en explore_data.py Puedo pasarla a utility
def pickle_to_dataframe(dataset):
    label_features = ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy',
                      'spectral_flux', 'spectral_rolloff', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6',
                      'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1', 'chroma_2',
                      'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10',
                      'chroma_11', 'chroma_12', 'chroma_std', 'harmonic_ratio', 'pitch']

    dataset_path = "data/" + dataset + "/" + dataset
    print("Loading data from " + dataset + " data set...")
    ds = pickle.load(open(dataset_path + '_db.p', 'rb'))
    print("Number of samples: " + str(len(ds.targets)))
    print("Loading features from file...\n")
    features = pickle.load(open(dataset_path + '_features.p', 'rb'))

    df_emotions = pd.Series(data=ds.targets, name="emotion", dtype="category")

    selected_emotion = "sadness"
    number_selected_emotion = ds.name_emotions.index(selected_emotion)
    df_emotions = df_emotions.apply(lambda i: "sadness" if i == number_selected_emotion else "other")

    mean_features = []
    for single_audio in features:
        mean_features.append(np.apply_along_axis(np.mean, 0, single_audio))
    df_features = pd.DataFrame.from_records(data=mean_features, columns=label_features)

    df_concat = pd.concat([df_features, df_emotions], axis=1)
    # df_concat["dataset"] = ds.name_dataset
    return df_concat


if __name__ == '__main__':
    dataset = "enterface-0.015-01234"
    df = pickle_to_dataframe(dataset)

    """
    df = df.drop(['mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6',
                      'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1', 'chroma_2',
                      'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10',
                      'chroma_11', 'chroma_12', 'chroma_std'], axis=1)
        label_features = ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy',
                      'spectral_flux', 'spectral_rolloff', 'harmonic_ratio', 'pitch']
    """

    x = df.drop(["emotion"], axis=1)
    y = df["emotion"]
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)

    clf = DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(xTrain, yTrain)
    yPred = clf.predict(xTest)

    print("Accuracy:", metrics.accuracy_score(yTest, yPred))

    label_features = ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy',
                      'spectral_flux', 'spectral_rolloff', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6',
                      'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1', 'chroma_2',
                      'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10',
                      'chroma_11', 'chroma_12', 'chroma_std', 'harmonic_ratio', 'pitch']

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=label_features, class_names=["other", "sadness"])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("data/tree_png/" + dataset + ".png")
    Image(graph.create_png())
