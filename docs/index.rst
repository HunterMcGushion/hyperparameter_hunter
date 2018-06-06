.. hyperparameter_hunter documentation master file, created by
   sphinx-quickstart on Mon Jun  4 20:28:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to hyperparameter_hunter's documentation!
=================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Environment
-----------

.. autoclass:: hyperparameter_hunter.environment.Environment
   :members:
   :show-inheritance:
   :noindex:

Experiment Execution
--------------------

.. autoclass:: hyperparameter_hunter.experiments.CrossValidationExperiment
   :members: __init__
   :show-inheritance:
   :noindex:

Hyperparameter Optimization
---------------------------

.. autoclass:: hyperparameter_hunter.optimization.BayesianOptimization
   :members: __init__, set_experiment_guidelines, go
   :show-inheritance:
   :noindex:

.. autoclass:: hyperparameter_hunter.optimization.GradientBoostedRegressionTreeOptimization
   :members: __init__, set_experiment_guidelines, go
   :show-inheritance:
   :noindex:

.. autoclass:: hyperparameter_hunter.optimization.RandomForestOptimization
   :members: __init__, set_experiment_guidelines, go
   :show-inheritance:
   :noindex:

.. autoclass:: hyperparameter_hunter.optimization.ExtraTreesOptimization
   :members: __init__, set_experiment_guidelines, go
   :show-inheritance:
   :noindex:

.. autoclass:: hyperparameter_hunter.optimization.DummySearch
   :members: __init__, set_experiment_guidelines, go
   :show-inheritance:
   :noindex:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
