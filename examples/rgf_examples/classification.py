from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter import BayesianOptimization, Real, Integer, Categorical
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from rgf import RGFClassifier

#################### Format DataFrame ####################
x, y = make_classification(n_samples=700, n_classes=2, shuffle=True, random_state=32)
train_df = pd.DataFrame(x, columns=range(x.shape[1]))
train_df["y"] = y

#################### Set Up Environment ####################
env = Environment(
    train_dataset=train_df,
    results_path="HyperparameterHunterAssets",
    target_column="y",
    metrics_map=["hamming_loss"],
    cross_validation_type=RepeatedStratifiedKFold,
    cross_validation_params=dict(n_repeats=2, n_splits=10, random_state=1337),
)

# Now that HyperparameterHunter has an active `Environment`, we can do two things:
#################### 1. Perform Experiments ####################
experiment = CVExperiment(
    model_initializer=RGFClassifier,
    model_init_params=dict(max_leaf=1000, algorithm="RGF", min_samples_leaf=10),
)

# And/or...
#################### 2. Hyperparameter Optimization ####################
optimizer = BayesianOptimization(iterations=10, random_state=42)
optimizer.set_experiment_guidelines(
    model_initializer=RGFClassifier,
    model_init_params=dict(
        max_leaf=1000,
        algorithm=Categorical(["RGF", "RGF_Opt", "RGF_Sib"]),
        l2=Real(0.01, 0.3),
        normalize=Categorical([True, False]),
        learning_rate=Real(0.3, 0.7),
        loss=Categorical(["LS", "Expo", "Log", "Abs"]),
    ),
)
optimizer.go()

# Notice, `optimizer` recognizes our earlier `experiment`'s hyperparameters fit inside the search
# space/guidelines set for `optimizer`.

# Then, when optimization is started, it automatically learns from `experiment`'s results
# - without any extra work for us!
