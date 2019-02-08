from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter import ExtraTreesOptimization, Real, Integer, Categorical
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_log_error
from rgf import RGFRegressor

#################### Format DataFrame ####################
data = load_diabetes()
train_df = pd.DataFrame(data=data.data, columns=data.feature_names)
train_df["progression"] = data.target

#################### Set Up Environment ####################
env = Environment(
    train_dataset=train_df,
    results_path="HyperparameterHunterAssets",
    target_column="progression",
    metrics=dict(msle=(mean_squared_log_error, "min")),
    cv_type="KFold",
    cv_params=dict(n_splits=10, random_state=42),
    runs=3,
)

# Now that HyperparameterHunter has an active `Environment`, we can do two things:
#################### 1. Perform Experiments ####################
experiment = CVExperiment(
    model_initializer=RGFRegressor,
    model_init_params=dict(max_leaf=2000, algorithm="RGF", min_samples_leaf=10),
)

# And/or...
#################### 2. Hyperparameter Optimization ####################
optimizer = ExtraTreesOptimization(iterations=30, random_state=42)
optimizer.set_experiment_guidelines(
    model_initializer=RGFRegressor,
    model_init_params=dict(
        max_leaf=2000,
        algorithm=Categorical(["RGF", "RGF_Opt", "RGF_Sib"]),
        l2=Real(0.01, 0.3),
        normalize=Categorical([True, False]),
        learning_rate=Real(0.3, 0.7),
        loss=Categorical(["LS", "Expo", "Log"]),
    ),
)
optimizer.go()

# Notice, `optimizer` recognizes our earlier `experiment`'s hyperparameters fit inside the search
# space/guidelines set for `optimizer`.

# Then, when optimization is started, it automatically learns from `experiment`'s results
# - without any extra work for us!
