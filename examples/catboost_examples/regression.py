from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter import DummySearch, Real, Integer, Categorical
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import explained_variance_score
from catboost import CatBoostRegressor

#################### Format DataFrame ####################
data = make_regression(n_samples=600, n_features=50, noise=0.1, random_state=42)
train_df = pd.DataFrame(data[0], columns=["x_{}".format(_) for _ in range(data[0].shape[1])])
train_df["target"] = data[1]

#################### Set Up Environment ####################
env = Environment(
    train_dataset=train_df,
    root_results_path="HyperparameterHunterAssets",
    metrics_map=dict(evs=explained_variance_score),
    cross_validation_type="KFold",
    cross_validation_params=dict(n_splits=3, shuffle=True, random_state=1337),
    runs=2,
)

# Now that HyperparameterHunter has an active `Environment`, we can do two things:
#################### 1. Perform Experiments ####################
# *Note: If this is your first HyperparameterHunter example, the CatBoost classification example may be a better starting point.*

# In this Experiment, we're also going to use `model_extra_params` to provide arguments to
# ... `CatBoostRegressor`'s `fit` method, just like we would if we weren't using HyperparameterHunter.

# We'll be using the `verbose` argument to print evaluations of our `CatBoostRegressor` every 50 iterations,
# ... and we'll also be using the dataset sentinels offered by `Environment`. You can read more about
# ... the exciting thing you can do with the `Environment` sentinels in the documentation and in the
# ... example dedicated to them. For now, though, we'll be using them to provide each fold's
# ... `env.validation_input`, and `env.validation_target` to `CatBoostRegressor.fit` via its `eval_set` argument.

# You could also easily add `CatBoostRegressor.fit`'s `early_stopping_rounds` argument to `model_extra_params["fit"]`
# ... to use early stopping, but doing so here with only `iterations=100` doesn't make much sense.
experiment = CVExperiment(
    model_initializer=CatBoostRegressor,
    model_init_params=dict(
        iterations=100,
        learning_rate=0.05,
        depth=5,
        bootstrap_type="Bayesian",
        save_snapshot=False,
        allow_writing_files=False,
    ),
    model_extra_params=dict(
        fit=dict(verbose=50, eval_set=[(env.validation_input, env.validation_target)])
    ),
)

# Notice above that CatBoost printed scores for our `eval_set` every 50 iterations just like we said
# ... in `model_extra_params["fit"]`; although, it made our results rather difficult to read, so
# ... we'll switch back to `verbose=False` during optimization.

# And/or...
#################### 2. Hyperparameter Optimization ####################
# Notice below that `optimizer` still recognizes the results of `experiment` as valid learning material even
# ... though their `verbose` values differ. This is because it knows that `verbose` has no effect on actual results.
optimizer = DummySearch(iterations=10, random_state=777)
optimizer.set_experiment_guidelines(
    model_initializer=CatBoostRegressor,
    model_init_params=dict(
        iterations=100,
        learning_rate=Real(0.001, 0.2),
        depth=Integer(3, 7),
        bootstrap_type=Categorical(["Bayesian", "Bernoulli"]),
        save_snapshot=False,
        allow_writing_files=False,
    ),
    model_extra_params=dict(
        fit=dict(verbose=False, eval_set=[(env.validation_input, env.validation_target)])
    ),
)
optimizer.go()

# Notice, `optimizer` recognizes our earlier `experiment`'s hyperparameters fit inside the search
# space/guidelines set for `optimizer`.

# Then, when optimization is started, it automatically learns from `experiment`'s results
# - without any extra work for us!
