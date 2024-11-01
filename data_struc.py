from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from config import Config


def split_data(data, target, data_name, seed, data_path=None):
    """
    Split the data into training and testing sets
    :param data: the data
    :param target: the target
    :param data_name: the name of the dataset
    :param seed: the random seed
    :param data_path: the path to the data
    """
    config = Config()
    if data_path is None:
        data_path = config.data_path
    df = pd.read_csv(os.path.join(data_path, config.data_info))
    specific_dataset = df[df["Dataset Name"] == data_name]
    test_size = specific_dataset["Test Size"].values[0]
    train_size = specific_dataset["Train Size"].values[0]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=seed)
    assert len(X_train) == train_size
    return X_train, X_test, y_train, y_test


def make_imbalance(X, y, sampling_ratio=None, minority_num=False):
    """
    Make the data imbalanced
    :param X: the data
    :param y: the target
    :param sampling_ratio: the sampling ratio
    """
    config = Config()
    if sampling_ratio is None:
        sampling_ratio = config.imbalance_ratio
    x_minority = X[y == 1]  # Minority class
    x_majority = X[y == 0]  # Majority class

    labels, counts = np.unique(y, return_counts=True)
    imbalance_ratio = counts[0] / counts[1]
    if imbalance_ratio > sampling_ratio:
        indices = np.arange(len(x_majority))  # 创建索引数组
        np.random.shuffle(indices)
        x_majority = x_majority[indices][:int(len(x_minority) * sampling_ratio)]

    else:
        indices = np.arange(len(x_minority))
        np.random.shuffle(indices)
        x_minority = x_minority[indices][:int(len(x_majority) // sampling_ratio)]

    y_majority = np.zeros(len(x_majority))
    y_minority = np.ones(len(x_minority))
    x_imbalanced = np.vstack((x_majority, x_minority))
    y_imbalanced = np.hstack((y_majority, y_minority))

    if minority_num:
        return x_imbalanced, y_imbalanced, len(x_minority)
    return x_imbalanced, y_imbalanced
