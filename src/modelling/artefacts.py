"""Module artefacts.py"""
import logging
import os
import pandas as pd
import tensorflow as tf

import src.elements.intermediary as itr
import src.elements.master as mr
import src.elements.sequences as sq
import src.functions.streams


class Artefacts:
    """
    Artefacts
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__arguments = arguments

        # Instances
        self.__streams = src.functions.streams.Streams()

    def __history(self, model: tf.keras.src.models.Sequential, path: str):
        """

        :param model:
        :param path:
        :return:
        """

        history = pd.DataFrame(data=model.history.history)

        return self.__streams.write(blob=history, path=os.path.join(path, 'history.csv'))

    # noinspection PyUnresolvedReferences
    def exc(self, model: tf.keras.src.models.Sequential, sequences: sq.Sequences, intermediary: itr.Intermediary, master: mr.Master) -> str:
        """

        :param model:
        :param sequences:
        :param intermediary:
        :param master:
        :return:
        """

        self.__history(model=model, path=master.path)






        return 'in progress'
