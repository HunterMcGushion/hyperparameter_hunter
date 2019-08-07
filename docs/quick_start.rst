Quick Start
***********
This section provides a jumping-off point for using HyperparameterHunter's main features.

Set Up an Environment
=====================

.. code-block:: python

    from hyperparameter_hunter import Environment, CVExperiment
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    data = load_breast_cancer()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df["target"] = data.target

    env = Environment(
	    train_dataset=df,
	    results_path="path/to/results/directory",
	    metrics=["roc_auc_score"],
	    cv_type=StratifiedKFold,
	    cv_params=dict(n_splits=5, shuffle=True, random_state=32)
    )

Individual Experimentation
--------------------------

.. code-block:: python

    experiment = CVExperiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(objective="reg:linear", max_depth=3, subsample=0.5)
    )

Hyperparameter Optimization
---------------------------

.. code-block:: python

    from hyperparameter_hunter import BayesianOptPro, Real, Integer, Categorical

    optimizer = BayesianOptPro(iterations=10, read_experiments=True)

    optimizer.forge_experiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(
            n_estimators=200,
            subsample=0.5,
            max_depth=Integer(2, 20),
            learning_rate=Real(0.0001, 0.5),
            booster=Categorical(["gbtree", "gblinear", "dart"]),
        )
    )

    optimizer.go()

Plenty of examples for different libraries, and algorithms, as well as more advanced HyperparameterHunter features can be found
in the `examples <https://github.com/HunterMcGushion/hyperparameter_hunter/blob/master/examples>`__ directory.
