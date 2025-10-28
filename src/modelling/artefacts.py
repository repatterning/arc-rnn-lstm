"""Module artefacts.py"""
import logging
import pandas as pd
import tensorflow as tf

import src.elements.intermediary as itr
import src.elements.master as mr
import src.elements.sequences as sq


class Artefacts:
    """
    Artefacts
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__arguments = arguments

    # noinspection PyUnresolvedReferences
    def exc(self, model: tf.keras.src.models.Sequential, sequences: sq.Sequences, intermediary: itr.Intermediary, master: mr.Master) -> str:
        """

        :param model:
        :param sequences:
        :param intermediary:
        :param master:
        :return:
        """

        history = pd.DataFrame(data=model.history.history)
        history.info()

        # Arguments
        logging.info(self.__arguments)

        # The training sets
        intermediary.training.info()
        master.training.info()

        return 'in progress'
