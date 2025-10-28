"""Module persist.py"""
import os

import pandas as pd

import config
import src.elements.partitions as pr
import src.functions.streams


class Persist:
    """
    Constructor
    """

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = config.Config()
        self.__streams = src.functions.streams.Streams()

    def __persist(self, blob: pd.DataFrame, path: str) -> str:
        """

        :param blob:
        :param path:
        :return:
        """

        return self.__streams.write(blob=blob, path=path)

    def exc(self, partition: pr.Partitions):
        """

        :param partition:
        :return:
        """


        os.path.join(
            self.__configurations.assets_, str(partition.catchment_id), str(partition.ts_id))
