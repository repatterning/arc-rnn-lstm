
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
        self.__arguments = arguments

        # Scaling Arguments
        self.__features = self.__arguments.get('scaling').get('features')

        # Modelling Arguments
        self.__fields, self.__targets, self.__disjoint = self.__get_modelling_arguments()

        # Renaming
        self.__rename = { arg: f'e_{arg}' for arg in self.__targets}

    def __get_modelling_arguments(self):
        """

        :return:
        """

        elements: dict = self.__arguments.get('modelling')

        fields: list = elements.get('fields')
        targets: list = elements.get('targets')

        # The variables present within the [input] fields, but not the targets
        disjoint: list = list(set(fields).difference(set(targets)))

        return fields, targets, disjoint

    def __restructure(self, inverse: np.ndarray) -> pd.DataFrame:
        """

        :param inverse:
        :return:
        """

        frame = pd.DataFrame()
        frame.loc[:, self.__features] = inverse
        frame.rename(columns=self.__rename, inplace=True)

        return frame

    def exc(self, x_matrix: np.ndarray, frame: pd.DataFrame):
        """

        :param x_matrix:
        :param frame:
        :return:
        """

        predictions: np.ndarray = self.__model.predict(x=x_matrix)

        # The inverse transform structure
        structure = frame.copy()[self.__disjoint][-x_matrix.shape[0]:]
        structure.loc[:, self.__targets] = predictions
        structure = structure.copy()[self.__features]

        # Inverting
        inverse: np.ndarray = self.__scaler.inverse_transform(structure.values)

        return self.__restructure(inverse=inverse.copy())
