"""Module sequencing.py"""
import numpy as np
import pandas as pd


class Sequencing:
    """
    Builds the modelling windows, i.e., sequences of historical data & corresponding target
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__n_sequence = arguments.get('n_sequence')
        self.__fields = arguments.get('modelling').get('fields')
        self.__targets = arguments.get('modelling').get('targets')

    def exc(self, blob: pd.DataFrame):
        """

        :param blob: A modelling data set
        :return:
        """

        data = blob.copy().loc[:, self.__fields]
        matrix = data.values
        __indices = [data.columns.get_loc(k) for k in self.__targets]

        __limit = self.__n_sequence

        x_matrix = []
        y_matrix = []
        for j in range(data.shape[0] - __limit):
            x_matrix.append(matrix[j:(j + __limit)])
            y_matrix.append(matrix[j + __limit][__indices])

        return np.array(x_matrix), np.array(y_matrix)
