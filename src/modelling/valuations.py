
import tensorflow as tf

import numpy as np
import pandas as pd
import sklearn

class Valuations:

    def __init__(self, model: tf.keras.src.models.Sequential, scaler: sklearn.preprocessing.MinMaxScaler, arguments: dict):
        """

        :param model:
        :param scaler:
        :param arguments:
        """

        self.__model = model
        self.__scaler = scaler

        # Arguments
        self.__m_arguments: dict = arguments.get('modelling')

        self.__rename = { arg: f'e_{arg}' for arg in self.__m_arguments.get('targets')}
        self.__disjoint = list(set(self.__m_arguments.get('fields')).difference(
            set(self.__m_arguments.get('targets'))))

    def exc(self, x_matrix: np.ndarray, frame: pd.DataFrame):
        """

        :param x_matrix:
        :param frame:
        :return:
        """

        predictions: np.ndarray = self.__model.predict(x=x_matrix)
