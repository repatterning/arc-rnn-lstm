"""Module artefacts.py"""
import os

import pandas as pd
import sklearn
import tensorflow as tf

import src.functions.streams
import src.functions.objects


class Artefacts:
    """
    Artefacts
    """

    def __init__(self, model: tf.keras.src.models.Sequential, scaler: sklearn.preprocessing.MinMaxScaler, arguments: dict, path: str):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__model = model
        self.__scaler = scaler
        self.__arguments = arguments
        self.__path = path

        # Instances
        self.__streams = src.functions.streams.Streams()
        self.__objects = src.functions.objects.Objects()

    def __scaling(self):
        """

        :return:
        """

        values = {
            'data_max_': list(self.__scaler.data_max_),
            'data_min_': list(self.__scaler.data_min_),
            'data_range_': list(self.__scaler.data_range_),
            'feature_names_in_': list(self.__scaler.feature_names_in_),
            'n_features_in_': self.__scaler.n_features_in_,
            'feature_names_out': list(self.__scaler.get_feature_names_out()),
            'n_samples_seen_': self.__scaler.n_samples_seen_
        }

        return self.__objects.write(nodes=values, path=os.path.join(self.__path, 'scaling.json'))

    def __history(self):
        """

        :return:
        """

        history = pd.DataFrame(data=self.__model.history.history)

        return self.__streams.write(blob=history, path=os.path.join(self.__path, 'history.csv'))

    def exc(self):

        self.__history()
