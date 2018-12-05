from hyperparameter_hunter import Environment, CrossValidationExperiment
from hyperparameter_hunter import ExtraTreesOptimization, Real, Integer, Categorical
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from lightgbm import LGBMRegressor

#################### Format DataFrame ####################
data = load_boston()
train_df = pd.DataFrame(data=data.data, columns=data.feature_names)
train_df["median_value"] = data.target

#################### Set Up Environment ####################
env = Environment(
    train_dataset=train_df,
    root_results_path="HyperparameterHunterAssets",
    target_column="median_value",
    metrics_map=dict(r2=r2_score),
    cross_validation_type=RepeatedKFold,
    cross_validation_params=dict(n_repeats=2, n_splits=5, random_state=42),
)

# Now that HyperparameterHunter has an active `Environment`, we can do two things:
#################### 1. Perform Experiments ####################
experiment = CrossValidationExperiment(
    model_initializer=LGBMRegressor,
    model_init_params=dict(boosting_type="gbdt", num_leaves=31, min_child_samples=5, subsample=0.5),
)

# And/or...
#################### 2. Hyperparameter Optimization ####################
optimizer = ExtraTreesOptimization(iterations=12, random_state=1337)
optimizer.set_experiment_guidelines(
    model_initializer=LGBMRegressor,
    model_init_params=dict(
        boosting_type=Categorical(["gbdt", "dart"]),
        num_leaves=Integer(10, 40),
        max_depth=-1,
        min_child_samples=5,
        subsample=Real(0.3, 0.7),
    ),
)
optimizer.go()

# Notice, `optimizer` recognizes our earlier `experiment`'s hyperparameters fit inside the search
# space/guidelines set for `optimizer`.

# Then, when optimization is started, it automatically learns from `experiment`'s results
# - without any extra work for us!
