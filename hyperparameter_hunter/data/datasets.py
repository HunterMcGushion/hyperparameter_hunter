##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.data.data_core import BaseDataset, NullDataChunk
from hyperparameter_hunter.data.data_chunks.input_chunks import (
    TrainInputChunk,
    OOFInputChunk,
    HoldoutInputChunk,
    TestInputChunk,
)
from hyperparameter_hunter.data.data_chunks.target_chunks import (
    TrainTargetChunk,
    OOFTargetChunk,
    HoldoutTargetChunk,
)
from hyperparameter_hunter.data.data_chunks.prediction_chunks import (
    OOFPredictionChunk,
    HoldoutPredictionChunk,
    TestPredictionChunk,
)


##################################################
# Datasets
##################################################
class TrainDataset(BaseDataset):
    _input_type: type = TrainInputChunk
    _target_type: type = TrainTargetChunk
    _prediction_type: type = NullDataChunk

    input: TrainInputChunk
    target: TrainTargetChunk
    prediction: NullDataChunk


class OOFDataset(BaseDataset):
    _input_type: type = OOFInputChunk
    _target_type: type = OOFTargetChunk
    _prediction_type: type = OOFPredictionChunk

    input: OOFInputChunk
    target: OOFTargetChunk
    prediction: OOFPredictionChunk


class HoldoutDataset(BaseDataset):
    _input_type: type = HoldoutInputChunk
    _target_type: type = HoldoutTargetChunk
    _prediction_type: type = HoldoutPredictionChunk

    input: HoldoutInputChunk
    target: HoldoutTargetChunk
    prediction: HoldoutPredictionChunk


class TestDataset(BaseDataset):
    _input_type: type = TestInputChunk
    _target_type: type = NullDataChunk
    _prediction_type: type = TestPredictionChunk

    input: TestInputChunk
    target: NullDataChunk
    prediction: TestPredictionChunk
