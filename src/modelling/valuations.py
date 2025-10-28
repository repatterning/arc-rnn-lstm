
import tensorflow as tf

import numpy as np

class Valuations:

    def __init__(self, model: tf.keras.src.models.Sequential, arguments: dict):

        self.__model = model
        self.__rename = { arg: f'e_{arg}' for arg in arguments.get('modelling').get('targets')}

    def exc(self, x_matrix: np.ndarray):

        predictions: np.ndarray = self.__model.predict(x=x_matrix)
