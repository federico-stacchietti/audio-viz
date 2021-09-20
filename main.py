import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder, StandardScaler

import cuml


def read_dataset(path: str, drop=True, process=False, encode_label=False,
                 oversample=None, min_sample=None) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    if drop:
        dataset.drop(dataset.columns[0], inplace=True, axis=1)

    if encode_label:
        X, y = dataset[dataset.columns[1:-1]].values.astype('float32'), dataset['label'].values
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y).astype('float32')
        y = y.reshape(y.shape[0], 1)
        values = np.hstack((X, y))
        dataset = pd.DataFrame(data=values, columns=dataset.columns[1:])
        dataset = dataset.sample(frac=1).reset_index(drop=True)

    if process:
        scaler = StandardScaler()
        X, y = scaler.fit_transform(dataset[dataset.columns[1:-1]].values.astype('float32')), \
               dataset['label'].values
        y = y.reshape(y.shape[0], 1)
        values = np.hstack((X, y))
        dataset = pd.DataFrame(data=values, columns=dataset.columns[1:])
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    if oversample is not None:
        strategy = {}
        labels_count = dataset['label'].value_counts()
        X, y = dataset[dataset.columns[1:-1]].values.astype('float32'), \
               dataset['label'].values
        for label, count in zip(labels_count.index.tolist(), labels_count.values):
            if count < min_sample:
                count = oversample
            strategy[label] = count
        oversample = SMOTE(sampling_strategy=strategy)
        X, y = oversample.fit_resample(X, y)
        y = y.reshape(y.shape[0], 1)

        values = np.hstack((X, y))

        dataset = pd.DataFrame(data=values, columns=dataset.columns[1:])
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    return dataset


if __name__ == '__main__':
    training_path = 'features_dataset/ae_features/train2/training_set.csv'
    test_path = 'features_dataset/ae_features/test2/test_set.csv'

    training_set = read_dataset(training_path, drop=True, process=True, encode_label=True,
                                oversample=500, min_sample=250)
    test_set = read_dataset(test_path, drop=True, process=True, encode_label=True)

    import plotly.express as px

    df = pd.concat([training_set, test_set])

    df = df.head(2000)

    X, y = df[df.columns[:-1]].values.astype('float32'), df['label'].values

    pca = cuml.PCA(n_components=10)
    X = pca.fit_transform(X)

    t_sne = cuml.TSNE(n_components=2, n_neighbors=120, perplexity=30)
    X = t_sne.fit_transform(X)

    y = y.reshape(y.shape[0], 1)

    values = np.hstack((X, y))

    dataset = pd.DataFrame(data=values, columns=['x', 'y', 'label'])

    print(dataset)

    fig = px.scatter(dataset, x='x', y='y', color='label')

    fig.show()