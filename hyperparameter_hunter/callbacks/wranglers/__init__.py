"""This module contains the final implementations of the three types of
:class:`~hyperparameter_hunter.callbacks.bases.BaseWranglerCallback` descendants. The callbacks
defined herein act as liaisons between the experiment and its datasets (the datasets' data chunks).
Each callback in the module is expected to be responsible for a specific descendant of
:class:`hyperparameter_hunter.data.data_core.BaseDataChunk`, which can be seen from the type
annotation at the forefront of each callback class for its "data"-prefixed attribute

Each callback in the module is actually pulling its dataset (and the appropriate data chunk) from
the experiment via its four dataset attributes:

* `data_train`: :class:`~hyperparameter_hunter.data.datasets.TrainDataset`
* `data_oof`: :class:`~hyperparameter_hunter.data.datasets.OOFDataset`
* `data_holdout`: :class:`~hyperparameter_hunter.data.datasets.HoldoutDataset`
* `data_test`: :class:`~hyperparameter_hunter.data.datasets.TestDataset`

Specifically, each callback herein is responsible for the data chunk denoted by the name of that
callback's immediate parent callback, which is one of the following:

* :class:`~hyperparameter_hunter.callbacks.bases.BaseInputWranglerCallback`
* :class:`~hyperparameter_hunter.callbacks.bases.BaseTargetWranglerCallback`
* :class:`~hyperparameter_hunter.callbacks.bases.BasePredictorCallback`

Related
-------
:mod:`hyperparameter_hunter.data`
    This module defines the data chunks (attributes of datasets), for which each callback defined in
    :mod:`~hyperparameter_hunter.callbacks.wranglers` is responsible. This responsibility is usually
    satisfied by simply invoking the correct callback method. However, occasionally a data chunk's
    callback method will require additional inputs. In these cases, the wrangler callbacks must
    ensure the proper arguments are provided"""
# TODO: Input chunks/wranglers probably unnecessary, since their 2 jobs are done by `CVExperiment`
