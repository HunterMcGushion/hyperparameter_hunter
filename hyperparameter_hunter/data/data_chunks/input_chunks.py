##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.data.data_core import BaseDataChunk


# TODO: Input chunks/wranglers probably unnecessary, since their 2 jobs are done by `CVExperiment`


class BaseInputChunk(BaseDataChunk):
    ...


class TrainInputChunk(BaseInputChunk):
    ...


class OOFInputChunk(BaseInputChunk):
    ...


class HoldoutInputChunk(BaseInputChunk):
    ...


class TestInputChunk(BaseInputChunk):
    ...
