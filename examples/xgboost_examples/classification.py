from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter import BayesianOptimization, Real, Integer, Categorical
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

#################### Format DataFrame ####################
data = load_breast_cancer()
train_df = pd.DataFrame(data.data, columns=data.feature_names)
train_df["diagnosis"] = data.target

#################### Set Up Environment ####################
env = Environment(
    train_dataset=train_df,
    results_path="HyperparameterHunterAssets",
    target_column="diagnosis",
    metrics_map=["roc_auc_score"],
    cross_validation_type=StratifiedKFold,
    cross_validation_params=dict(n_splits=10, shuffle=True, random_state=32),
)

# Now that HyperparameterHunter has an active `Environment`, we can do two things:
#################### 1. Perform Experiments ####################
experiment = CVExperiment(
    model_initializer=XGBClassifier,
    model_init_params=dict(objective="reg:linear", max_depth=3, n_estimators=100, subsample=0.5),
    model_extra_params=dict(
        fit=dict(
            eval_set=[
                (env.train_input, env.train_target),
                (env.validation_input, env.validation_target),
            ],
            early_stopping_rounds=5,
            eval_metric="mae",
        )
    ),
)

# And/or...
#################### 2. Hyperparameter Optimization ####################
optimizer = BayesianOptimization(iterations=30, random_state=1337)
optimizer.set_experiment_guidelines(
    model_initializer=XGBClassifier,
    model_init_params=dict(
        objective="reg:linear",
        max_depth=Integer(2, 20),
        learning_rate=Real(0.0001, 0.5),
        subsample=0.5,
        booster=Categorical(["gbtree", "dart"]),
    ),
    model_extra_params=dict(
        fit=dict(
            eval_set=[
                (env.train_input, env.train_target),
                (env.validation_input, env.validation_target),
            ],
            early_stopping_rounds=5,
            eval_metric=Categorical(["auc", "mae"]),
        )
    ),
)
optimizer.go()

# Notice, `optimizer` recognizes our earlier `experiment`'s hyperparameters fit inside the search
# space/guidelines set for `optimizer`.

# Then, when optimization is started, it automatically learns from `experiment`'s results
# - without any extra work for us!
