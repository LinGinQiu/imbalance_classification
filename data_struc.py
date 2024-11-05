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

def shuffle_data(X_train, X_test, y_train, y_test, seed):
    """
    Shuffle the data, concatenate the train data and target, and shuffle them
    :param X_train: the training data
    :param X_test: the testing data
    :param y_train: the training target
    :param y_test: the testing target
    :param seed: the random seed
    """
    x_train_majority = X_train[y_train == 0]
    x_train_minority = X_train[y_train == 1]
    x_test_majority = X_test[y_test == 0]
    x_test_minority = X_test[y_test == 1]

    x_majority = np.vstack((x_train_majority, x_test_majority))
    x_minority = np.vstack((x_train_minority, x_test_minority))

    index_majority = np.arange(len(x_majority))
    index_minority = np.arange(len(x_minority))
    np.random.seed(seed)
    np.random.shuffle(index_majority)
    np.random.shuffle(index_minority)

    x_train_majority_new = x_majority[index_majority][:len(x_train_majority)]
    x_test_majority_new = x_majority[index_majority][len(x_train_majority):]
    x_train_minority_new = x_minority[index_minority][:len(x_train_minority)]
    x_test_minority_new = x_minority[index_minority][len(x_train_minority):]

    X_train_new = np.vstack((x_train_majority_new, x_train_minority_new))
    y_train_new = np.hstack((np.zeros(len(x_train_majority_new)), np.ones(len(x_train_minority_new))))
    X_test_new = np.vstack((x_test_majority_new, x_test_minority_new))
    y_test_new = np.hstack((np.zeros(len(x_test_majority_new)), np.ones(len(x_test_minority_new))))

    index_train = np.arange(len(X_train_new))
    index_test = np.arange(len(X_test_new))
    np.random.shuffle(index_train)
    np.random.shuffle(index_test)

    X_train_new = X_train_new[index_train]
    y_train_new = y_train_new[index_train]
    X_test_new = X_test_new[index_test]
    y_test_new = y_test_new[index_test]

    return X_train_new, X_test_new, y_train_new, y_test_new



def make_imbalance(X, y, sampling_ratio=None, minority_num=False):
    """
    Make the data imbalanced
    :param X: the data
    :param y: the target
    :param sampling_ratio: the sampling ratio
    """
    if sampling_ratio is None:
        return X, y

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
        minority_num = int(len(x_majority) // sampling_ratio)
        if minority_num <= 1:
            minority_num = 1
        x_minority = x_minority[indices][:minority_num]

    y_majority = np.zeros(len(x_majority))
    y_minority = np.ones(len(x_minority))
    x_imbalanced = np.vstack((x_majority, x_minority))
    y_imbalanced = np.hstack((y_majority, y_minority))

    index_shuffle = np.arange(len(x_imbalanced))
    np.random.shuffle(index_shuffle)
    x_imbalanced = x_imbalanced[index_shuffle]
    y_imbalanced = y_imbalanced[index_shuffle]

    if minority_num:
        return x_imbalanced, y_imbalanced, len(x_minority)
    return x_imbalanced, y_imbalanced
