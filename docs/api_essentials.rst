HyperparameterHunter API Essentials
***********************************
This section exposes the API for all the HyperparameterHunter functionality that will be necessary for most users.

Environment
===========

.. autoclass:: hyperparameter_hunter.environment.Environment
   :noindex:
   :members:

Experiment Execution
====================

.. autoclass:: hyperparameter_hunter.experiments.CVExperiment
   :noindex:
   :members: __init__

Hyperparameter Optimization
===========================

.. autoclass:: hyperparameter_hunter.optimization.BayesianOptPro
   :noindex:

   .. automethod:: __init__
      :noindex:
   .. automethod:: set_experiment_guidelines
      :noindex:
   .. automethod:: go
      :noindex:

----------------------------------------

.. autoclass:: hyperparameter_hunter.optimization.GradientBoostedRegressionTreeOptPro
   :noindex:

   .. automethod:: __init__
      :noindex:
   .. automethod:: set_experiment_guidelines
      :noindex:
   .. automethod:: go
      :noindex:

----------------------------------------

.. autoclass:: hyperparameter_hunter.optimization.RandomForestOptPro
   :noindex:

   .. automethod:: __init__
      :noindex:
   .. automethod:: set_experiment_guidelines
      :noindex:
   .. automethod:: go
      :noindex:

----------------------------------------

.. autoclass:: hyperparameter_hunter.optimization.ExtraTreesOptPro
   :noindex:

   .. automethod:: __init__
      :noindex:
   .. automethod:: set_experiment_guidelines
      :noindex:
   .. automethod:: go
      :noindex:

----------------------------------------

.. autoclass:: hyperparameter_hunter.optimization.DummyOptPro
   :noindex:

   .. automethod:: __init__
      :noindex:
   .. automethod:: set_experiment_guidelines
      :noindex:
   .. automethod:: go
      :noindex:

Extras
======

.. autoclass:: hyperparameter_hunter.feature_engineering.FeatureEngineer
   :noindex:
   :members: __init__

----------------------------------------

.. autoclass:: hyperparameter_hunter.feature_engineering.EngineerStep
   :noindex:
   :members: __init__

----------------------------------------

.. autofunction:: hyperparameter_hunter.callbacks.bases.lambda_callback
   :noindex:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
