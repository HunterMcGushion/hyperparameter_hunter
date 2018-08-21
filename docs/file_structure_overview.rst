File Structure Overview
***********************
This section is an overview of the result file structure created and updated when ``Experiment``\s are completed.

HyperparameterHunterAssets/
===========================

* Contains one file (**'Heartbeat.log'**), and four subdirectories (**'Experiments/'**, **'KeyAttributeLookup/'**,
  **'Leaderboards/'**, and **'TestedKeys/'**).
* **'Heartbeat.log'** is the log file for the current/most recently executed ``Experiment``. It will look very much like the
  printed output of ``CrossValidationExperiment``, with some additional debug messages thrown in. When the ``Experiment`` is
  completed, a copy of this file is saved as the ``Experiment``'s own Heartbeat file, which will be discussed below.

/**Experiments/**
-----------------
Contains up to six different subdirectories. The files contained in each of the subdirectories all follow the same naming
convention: they are named after the ``Experiment``'s randomly-generated UUID. The subdirectories are as follows:

----

1) /Descriptions/
~~~~~~~~~~~~~~~~~
Contains a .json file for each completed ``Experiment``, describing all critical (and some extra) information about the
``Experiment``'s results. Such information includes, but is certainly not limited to: keys, algorithm/library name, final scores,
model_initializer hash, hyperparameters, cross experiment parameters, breakdown of times elapsed, start/end datetimes,
breakdown of evaluations over runs/folds/reps, source script name, platform, and additional notes. This file is meant to give you
all the details you need regarding an ``Experiment``'s results and the conditions that led to those results.

2) /Heartbeats/
~~~~~~~~~~~~~~~
Contains a .log file for each completed ``Experiment`` that is created by copying the aforementioned
**'HyperparameterHunterAssets/Heartbeat.log'** file. This file is meant to give you a record of what exactly the ``Experiment``
was experiencing along the course of its existence. This can be useful if you need to verify questionable results, or check for
error/warning/debug messages that might not have been noticed before.

3) /PredictionsOOF/
~~~~~~~~~~~~~~~~~~~
Contains a .csv file for each completed ``Experiment``, containing out-of-fold predictions for the ``train_dataset`` provided to
``Environment``. If ``Environment`` is given a ``runs`` value > 1, or if a repeated cross-validation scheme is provided (like
sklearn's ``RepeatedKFold`` or ``RepeatedStratifiedKFold``), then OOF predictions will be averaged according to the number of
runs and repetitions. An extended discussion of this file's uses probably isn't necessary, but just some of the things you might
want it for include: testing the performance of ensembled models via their prediction files, or calculating other metric values,
if, for example, we wanted an F1 score, or simple accuracy after the ``Experiment`` had finished, instead of the ROC-AUC score we
told the ``Environment`` we wanted. Note that if we knew ahead of time we wanted all three of these metrics, we could have easily
given the ``Environment`` all three (or any other number of metrics) at its initialization. See the 'custom_metrics_example.py'
example script for more details on advanced metrics specifications.

4) /PredictionsHoldout/
~~~~~~~~~~~~~~~~~~~~~~~
This subdirectory's file structure is pretty much identical to **'PredictionsOOF/'** and is populated when we use
``Environment``'s ``holdout_dataset`` kwarg to provide a holdout DataFrame, a filepath to one, or a callable to extract a
``holdout_dataset`` from our ``train_dataset``. Additionally, if a ``holdout_dataset`` is provided, the provided metrics will be
calculated for it as well (unless you tell it otherwise).

5) /PredictionsTest/
~~~~~~~~~~~~~~~~~~~~
This subdirectory is much like **'PredictionsOOF/'** and **'PredictionsHoldout/'**. It is populated when we use ``Environment``'s
``test_dataset`` kwarg to provide a test DataFrame, or a filepath to one. It may be worth noting that the major difference
between ``test_dataset`` and its counterparts (``train_dataset``, and ``holdout_dataset``) is that test predictions are not
evaluated because it is the nature of the ``test_dataset`` to have unknown targets.

6) /ScriptBackups/
~~~~~~~~~~~~~~~~~~
Contains a .py file for each completed ``Experiment`` that is an exact copy of the script executed that led to the instantiation
of the ``Experiment``. These files exist primarily to assist in "oh shit" moments where you have no idea how to recreate an
``Experiment``. 'script_backup' is blacklisted by default when executing a hyperparameter ``OptimizationProtocol``, as all
experiments would be created by the same file.

----

/**KeyAttributeLookup/**
------------------------

* This directory stores any complex-typed ``Environment`` parameters and hyperparameters, as well as the hashes with which those
  complex objects are associated.
* Specifically, this directory is concerned with any python classes, or callables, or DataFrames you may provide, and will create
  a the appropriate file or directory to properly store the object.

    * If a class is provided (as is the case with ``cross_validation_type``, and ``model_initializer``), the Shelve and Dill
      libraries are used to pickle a copy of the class, linked to the class's hash as its key.
    * If a defined function, or a lambda is provided (as is the case with ``prediction_formatter``, which is an optional
      ``Environment`` kwarg), a .json file entry is created linking the callable's hash to its source code saved as a string,
      which can be recreated using Python's exec function.
    * If a Pandas DataFrame is provided (as is the case with ``train_dataset``, and its holdout and test counterparts), the
      process is slightly different. Rather than naming a file after the complex-typed attribute (as in the first two types), a
      directory is named after the attribute, hence the **'HyperparameterHunterAssets/KeyAttributeLookup/train_dataset/'**
      directory. Then, .csv files are added to the corresponding directory, which are named after the DataFrame's hash, and
      which contain the DataFrame itself.

* Entries in the **'KeyAttributeLookup/'** directory are created on an as-needed basis.

    * This means that you may see entries named after attributes other than those shown in this example along the course of your
      own project.
    * They are created whenever ``Environment``\s or ``Experiment``\s are provided arguments too complex to neatly display in the
      ``Experiment``'s **'Descriptions/'** entry file.
    * Some other complex attributes you may come across that are given **'KeyAttributeLookup/'** entries include: custom metrics
      provided via ``Environment``'s ``metrics_map`` and ``metrics_params`` kwargs, and Keras Neural Network ``callbacks`` and
      ``build_fn``\s.

/**Leaderboards/**
------------------
* At the time of this documentation's writing, this directory contains only one file: **'GlobalLeaderboard.csv'**; although, more
  are on the way to assist you in comparing the performance of different ``Experiment``\s, and they should be similar in structure
  to this one.
* **'GlobalLeaderboard.csv'** is a DataFrame containing one row for every completed ``Experiment``
* It has a column for every final metric evaluation performed, as well as the following columns: 'experiment_id',
  'hyperparameter_key', 'cross_experiment_key', and 'algorithm_name'
* Rows are sorted in descending order according to the first metric provided, and will prioritize OOF evaluations before holdout
  evaluations if both are given.
* If an ``Experiment`` does not have a particular evaluation, the ``Experiment`` row's value for that column will be null.

    * This can happen if new metrics are specified, which were not recorded for earlier experiments, or if a ``holdout_dataset``
      is provided to later ``Experiment``\s that earlier ones did not have.

/**TestedKeys/**
----------------
* This directory contains a .json file named for every unique ``cross_experiment_key`` encountered.
* Each .json file contains a dictionary, whose keys are the ``hyperparameter_key``\s that have been tested in conjunction with
  the ``cross_experiment_key`` for which the containing file is named.
* The value of each of these keys is a list of strings, in which each string is an ``experiment_id``, denoting an ``Experiment``
  that was conducted with the hyperparameters symbolized by that list's key, and an ``Environment``, whose cross-experiment
  parameters are symbolized by the name of the containing file.

    * The values are lists in order to accommodate ``Experiment``\s that are intentionally duplicated.







