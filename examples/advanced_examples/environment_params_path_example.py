from hyperparameter_hunter import Environment, CrossValidationExperiment
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def _execute():
    # To start, take a look at "examples/environment_params.json" - This is the file we're giving our Environment below
    # In this file, we can define a bunch of default Environment parameters that we don't want to always explicitly provide

    # It works really well for things that won't be changing often, like the following:
    # - `root_results_path`, which we probably never want to change, so all our results go to one place;
    # - `target_column`, which will probably be a constant for your data
    # - `metrics_map`, if you're not using any fancy metrics, and you already know what you want
    # - `file_blacklist`, if you're angry at me for adding that one result file that's always useless
    # Other parameters, whose default values you may want to change

    env = Environment(
        train_dataset=get_breast_cancer_data(),  # If your dataset is a str path, you can even add it to environment_params
        environment_params_path="./environment_params.json",  # Use this file for parameters not explicitly given
        cross_validation_params=dict(
            n_splits=5, shuffle=True, random_state=32
        ),  # Here we decide to override our default values
    )

    print(env.root_results_path)
    print(env.target_column)
    print(env.metrics_map)
    print(env.cross_validation_type)
    print(env.runs)
    print(env.file_blacklist)  # This includes some other values too, but you can ignore them
    # All of the above are from `environment_params_path`
    print(
        env.cross_validation_params
    )  # This is the value we provided above, rather than our `environment_params_path` default

    experiment = CrossValidationExperiment(
        model_initializer=KNeighborsClassifier, model_init_params={}
    )

    # We can see in the console that we're definitely evaluating both 'roc_auc_score', and 'f1_score', and we're doing 3 runs
    # We only did 5-fold cross-validation, as expected because we override our default value
    # And, notice there's no log saying the Heartbeat file was saved
    # We can also check "HyperparameterHunterAssets/Experiments/Heartbeats" for the experiment we just ran, and there's nothing!


if __name__ == "__main__":
    _execute()
