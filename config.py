"""
Module config.py
"""
import os


class Config:
    """
    Description
    -----------

    A class for configurations
    """

    def __init__(self) -> None:
        """
        <b>Notes</b><br>
        ------<br>

        Variables denoting a path - including or excluding a filename - have an underscore suffix; this suffix is
        excluded for names such as warehouse, storage, depository, *key, etc.<br><br>

        """

        '''
        Keys
        '''
        self.architecture = 'arc-rnn-lstm'
        self.s3_parameters_key = 's3_parameters.yaml'
        self.arguments_key = f'artefacts/architecture/{self.architecture}/arguments.json'
        self.metadata = 'artefacts/metadata.json'

        '''
        Project Metadata
        '''
        self.project_tag = 'hydrography'
        self.project_key_name = 'HydrographyProject'

        '''
        Local Paths
        '''
        sections = ['assets', self.architecture]
        self.warehouse: str = os.path.join(os.getcwd(), 'warehouse')
        self.assets_ = os.path.join(self.warehouse, *sections)

        '''
        Cloud
        '''
        self.prefix = '/'.join(sections)
