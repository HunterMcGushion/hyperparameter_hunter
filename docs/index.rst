.. hyperparameter_hunter documentation master file, created by
   sphinx-quickstart on Mon Jun  4 20:28:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HyperparameterHunter Documentation
**********************************

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Environment
===========

.. autoclass:: hyperparameter_hunter.environment.Environment
   :noindex:
   :members:

Experiment Execution
====================

.. autoclass:: hyperparameter_hunter.experiments.CrossValidationExperiment
   :noindex:
   :members: __init__

Hyperparameter Optimization
===========================

.. autoclass:: hyperparameter_hunter.optimization.BayesianOptimization
   :noindex:

   .. automethod:: __init__
   .. automethod:: set_experiment_guidelines
   .. automethod:: go

----------------------------------------

.. autoclass:: hyperparameter_hunter.optimization.GradientBoostedRegressionTreeOptimization
   :noindex:

   .. automethod:: __init__
   .. automethod:: set_experiment_guidelines
   .. automethod:: go

----------------------------------------

.. autoclass:: hyperparameter_hunter.optimization.RandomForestOptimization
   :noindex:

   .. automethod:: __init__
   .. automethod:: set_experiment_guidelines
   .. automethod:: go

----------------------------------------

.. autoclass:: hyperparameter_hunter.optimization.ExtraTreesOptimization
   :noindex:

   .. automethod:: __init__
   .. automethod:: set_experiment_guidelines
   .. automethod:: go

----------------------------------------

.. autoclass:: hyperparameter_hunter.optimization.DummySearch
   :noindex:

   .. automethod:: __init__
   .. automethod:: set_experiment_guidelines
   .. automethod:: go


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
