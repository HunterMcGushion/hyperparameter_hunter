HyperparameterHunter API Essentials
***********************************

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
