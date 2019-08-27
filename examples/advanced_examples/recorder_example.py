from hyperparameter_hunter import Environment, BayesianOptPro, Real, Integer, Categorical
from hyperparameter_hunter.io.recorders import UnsortedIDLeaderboardRecorder, BaseRecorder
from hyperparameter_hunter.utils.file_utils import make_dirs, read_json
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import yaml


#################### Format DataFrame ####################
data = load_breast_cancer()
train_df = pd.DataFrame(data.data, columns=data.feature_names)
train_df["diagnosis"] = data.target

#################### Set Up Environment ####################
# We'll set aside our `Environment` arguments, so we can easily reuse them later
env_kwargs = dict(
    train_dataset=train_df,
    results_path="HyperparameterHunterAssets",
    target_column="diagnosis",
    metrics=["roc_auc_score"],
    cv_type=StratifiedKFold,
    cv_params=dict(n_splits=10, shuffle=True, random_state=32),
)


# `experiment_recorders` works the same way with Experiments, so we'll just do optimization
#################### Hyperparameter Optimization ####################
# We'll set up a helper function, so we can easily re-run optimization with different Environments
def do_optimization():
    optimizer = BayesianOptPro(iterations=5, random_state=1337)
    optimizer.forge_experiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(
            objective="reg:linear",
            max_depth=Integer(2, 20),
            learning_rate=Real(0.0001, 0.5),
            subsample=0.5,
            booster=Categorical(["gbtree", "dart"]),
        ),
    )
    optimizer.go()


# We'll start with a normal `Environment` for comparison, using only the `env_kwargs` define above
env_0 = Environment(**env_kwargs)

do_optimization()


#################### Set Up Second Environment ####################
# Now, we'll make a new `Environment` that uses `experiment_recorders`
env_1 = Environment(
    **env_kwargs,
    # Here, we'll add an extra recorder defined in `hyperparameter_hunter.recorders`
    # This class adds a new leaderboard file that sorts entries according to all non-id columns
    # You can take a look at its implementation, but we're using it here to show what needs to be
    # ... provided to use `experiment_recorders`. We'll make our own custom recorder later
    experiment_recorders=[
        (UnsortedIDLeaderboardRecorder, "Leaderboards/UnsortedIDLeaderboard.csv")
    ],
    # `experiment_recorders` (if given) should be a list containing one tuple for each recorder
    # The first value in the tuple is our new class that descends from `recorders.BaseRecorder`
    # The second value is a str path relative to the `Environment`'s `results_path` that
    # ... tells our recorder where it should save the new result file
)

do_optimization()

# TODO: Look at directory contents and see new leaderboard


#################### Make Custom Recorder ####################
# The following is a BRIEF overview for implementing a custom recorder. Please see the documentation
# ... of `recorders.BaseRecorder` and its abstract methods outlined below for more essential info

# Custom recorders must subclass `recorders.BaseRecorder`, and implement these four abstract items:
# 1) `result_path_key`: Key location of the recorder's target result path - Used by `file_blacklist`
# 2) `required_attributes`: Experiment attributes used for result building - Exposed via `self`
# 3) `format_result`: Method to set `self.result` to the formatted object that will be saved
# 4) `save_result`: Method to save/update the result file (`self.result_path`) with `self.result`

# Additionally, there are two useful `BaseRecorder` attributes you should use:
# 1) `result_path`: Str directory in which our result should be saved - Concatenation of
# ... `Environment.results_path` and the second value given in the `experiment_recorders` tuple
# 2) `result`: (Optional) object we can set in `format_result` and use in `save_result`
class YAMLDescriptionRecorder(BaseRecorder):
    # This recorder class is just going to read the Experiment's default JSON description, then
    # ... write the object to a YAML file named for the "experiment_id"
    result_path_key = "yaml_description"
    required_attributes = ["result_paths", "experiment_id"]
    # Our `required_attributes` includes "result_paths", so we can access the saved JSON description
    # We're also including "experiment_id" (a common requirement) to use as the file's name

    def format_result(self):
        pass

    # Normally, we would define `result` in `format_result`, but we need to do that in `save_result`
    # ... so we'll leave `format_result` empty.
    # The reason for breaking this convention is that the JSON description file doesn't exist at the
    # ... time `format_result` is called because all recorders have their `format_result` methods
    # ... called, then they all have their `save_result` methods called
    # This is an odd example that calls for breaking the convention, which is rarely necessary

    def save_result(self):
        self.result = read_json(f"{self.result_paths['description']}/{self.experiment_id}.json")

        make_dirs(self.result_path, exist_ok=True)
        with open(f"{self.result_path}/{self.experiment_id}.yml", "w+") as f:
            yaml.dump(self.result, f, default_flow_style=False, width=200)


# Note in `save_result`, we simply set `result` to the content of the JSON file named for
# ... `experiment_id` in the `result_paths` directory for "description"s
# Then we open a YAML file also named for `experiment_id` in the `result_path` dir and dump `result`
# Remember that `result_path` is a special attribute set for us, as mentioned earlier

# This produces a file structure that mirrors the "Descriptions" directory, except with YAML files


#################### Set Up Third Environment ####################
# Now we can create a new `Environment` and add `YAMLDescriptionRecorder` to `experiment_recorders`
# Take special note of the dirpath that is the second value of our new tuple - This is appended to
# ... our `Environment.results_path` to create the special `result_path` attribute we used in
# ... our `YAMLDescriptionRecorder`'s `save_result` method
env_2 = Environment(
    **env_kwargs,
    experiment_recorders=[
        (UnsortedIDLeaderboardRecorder, "Leaderboards/UnsortedIDLeaderboard.csv"),
        (YAMLDescriptionRecorder, "Experiments/YAMLDescriptions"),
    ],
)

do_optimization()

# TODO: Look at YAMLDescriptions directory contents
