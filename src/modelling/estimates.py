
import tensorflow as tf

import src.elements.intermediary as itr
import src.elements.master as mr
import src.elements.sequences as sq
import src.modelling.valuations


class Estimates:
    """
    Estimates
    """

    def __init__(self, arguments: dict):
        """

        :param arguments: A set of arguments vis-à-vis calculation & storage objectives.
        """

        self.__arguments = arguments

    def exc(self, model: tf.keras.src.models.Sequential, sequences: sq.Sequences, intermediary: itr.Intermediary, master: mr.Master) -> str:
        """

        :param model:
        :param sequences:
        :param intermediary:
        :param master:
        :return:
        """

        valuations = src.modelling.valuations.Valuations(model=model, scaler=intermediary.scaler, arguments=self.__arguments)

        return 'in progress'
