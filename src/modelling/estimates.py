"""Module estimates.py"""
import tensorflow as tf

import pandas as pd


class Estimates:
    """
    Estimates
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__arguments = arguments

    def exc(self, model: tf.keras.src.models.Sequential):

        history = pd.DataFrame(data=model.history.history)
        history.info()


