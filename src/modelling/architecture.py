"""Module architecture.py"""

import numpy as np
import tensorflow as tf

import src.modelling.sequencing
import src.elements.intermediary as itr


class Architecture:
    """
    Architecture
    """

    def __init__(self, arguments: dict):
        """

        :param arguments:
        """

        self.__arguments = arguments
        self.__patience = self.__arguments.get('modelling').get('patience')
        self.__epochs = self.__arguments.get('modelling').get('epochs')
        self.__batch_size = self.__arguments.get('modelling').get('batch_size')

    def __model(self, x_tr: np.ndarray, y_tr: np.ndarray):
        """

        :param x_tr:
        :param y_tr:
        :return:
        """

        architecture = tf.keras.models.Sequential()
        architecture.add(tf.keras.layers.Input(shape=(x_tr.shape[1], x_tr.shape[2])))
        architecture.add(tf.keras.layers.LSTM(units=128, return_sequences=True))
        architecture.add(tf.keras.layers.LSTM(units=64, return_sequences=False))
        architecture.add(tf.keras.layers.Dense(units=1))

        # loss w.r.t. training data
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=self.__patience, mode='min')

        architecture.compile(
            loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])

        architecture.fit(
            x=x_tr, y=y_tr, epochs=self.__epochs, batch_size=self.__batch_size, callbacks=[early_stopping])

        return architecture

    def exc(self, intermediary: itr.Intermediary):
        """

        :param intermediary:
        :return:
        """

        seq = src.modelling.sequencing.Sequencing(arguments=self.__arguments)
        x_tr, y_tr = seq.exc(blob=intermediary.training)

        return self.__model(x_tr=x_tr, y_tr=y_tr)
