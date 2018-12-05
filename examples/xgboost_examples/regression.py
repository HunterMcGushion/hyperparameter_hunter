from hyperparameter_hunter import Environment, CrossValidationExperiment
from hyperparameter_hunter import GBRT, Real, Integer, Categorical
import pandas as pd
from sklearn.datasets import load_diabetes
from xgboost import XGBRegressor

#################### Format DataFrame ####################
data = load_diabetes()
train_df = pd.DataFrame(data=data.data, columns=data.feature_names)
train_df["progression"] = data.target

#################### Set Up Environment ####################
env = Environment(
    train_dataset=train_df,
    root_results_path="HyperparameterHunterAssets",
    target_column="progression",
    metrics_map=["mean_absolute_error"],
    cross_validation_type="KFold",
    cross_validation_params=dict(n_splits=12, shuffle=True, random_state=32),
    runs=2,
)

# Now that HyperparameterHunter has an active `Environment`, we can do two things:
#################### 1. Perform Experiments ####################
experiment = CrossValidationExperiment(
    model_initializer=XGBRegressor,
    model_init_params=dict(max_depth=4, n_estimators=400, subsample=0.5),
    model_extra_params=dict(fit=dict(eval_metric="mae")),
)

# And/or...
#################### 2. Hyperparameter Optimization ####################
optimizer = GBRT(iterations=20, random_state=32)
optimizer.set_experiment_guidelines(
    model_initializer=XGBRegressor,
    model_init_params=dict(
        max_depth=Integer(2, 20),
        n_estimators=Integer(100, 900),
        learning_rate=Real(0.0001, 0.5),
        subsample=0.5,
        booster=Categorical(["gbtree", "gblinear"]),
    ),
    model_extra_params=dict(fit=dict(eval_metric=Categorical(["rmse", "mae"]))),
)
optimizer.go()

# Notice, `optimizer` recognizes our earlier `experiment`'s hyperparameters fit inside the search
# space/guidelines set for `optimizer`.

# Then, when optimization is started, it automatically learns from `experiment`'s results
# - without any extra work for us!
