Quick Start
***********
This section provides a jumping-off point for using HyperparameterHunter's main features.

Set Up an Environment
=====================

.. code-block:: python

    from hyperparameter_hunter import Environment, CrossValidationExperiment
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    data = load_breast_cancer
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df['target'] = data.target

    env = Environment(
	    train_dataset=df,
	    root_results_path='path/to/results/directory',
	    metrics_map=['roc_auc_score'],
	    cross_validation_type=StratifiedKFold,
	    cross_validation_params=dict(n_splits=5, shuffle=2, random_state=32)
    )

Individual Experimentation
--------------------------

.. code-block:: python

    experiment = CrossValidationExperiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(objective='reg:linear', max_depth=3, subsample=0.5)
    )

Hyperparameter Optimization
---------------------------

.. code-block:: python

    from hyperparameter_hunter import BayesianOptimization, Real, Integer, Categorical

    optimizer = BayesianOptimization(
        iterations=100, read_experiments=True, dimensions=[
            Integer(name='max_depth', low=2, high=20),
            Real(name='learning_rate', low=0.0001, high=0.5),
            Categorical(name='booster', categories=['gbtree', 'gblinear', 'dart'])
        ]
    )

    optimizer.set_experiment_guidelines(
        model_initializer=XGBClassifier,
        model_init_params=dict(n_estimators=200, subsample=0.5, learning_rate=0.1)
    )
    optimizer.go()

Plenty of examples for different libraries, and algorithms, as well as more advanced HyperparameterHunter features can be found
in the `examples <https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples>`__ directory.
