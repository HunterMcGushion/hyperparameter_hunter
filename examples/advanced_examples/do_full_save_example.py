from hyperparameter_hunter import Environment, CrossValidationExperiment
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier


def do_full_save(experiment_result):
    """This is a simple check to see if the final OOF ROC-AUC score is above 0.75. If it is, we return True; otherwise, we return
    False. As input, your do_full_save functions should expect an Experiment's result dictionary. This is actually the dictionary
    that gets saved as the Experiment's "description" file, so for more information on what's in there, look at any description
    file or see :attr:`hyperparameter_hunter.recorders.DescriptionRecorder.result` (the object passed to `do_full_save`)"""
    return experiment_result["final_evaluations"]["oof"]["roc_auc_score"] > 0.75


def execute():
    env = Environment(
        train_dataset=get_toy_classification_data(),
        root_results_path="HyperparameterHunterAssets",
        metrics_map=["roc_auc_score"],
        cross_validation_type=RepeatedStratifiedKFold,
        cross_validation_params=dict(n_splits=3, n_repeats=2, random_state=32),
        do_full_save=do_full_save,
    )

    experiment_0 = CrossValidationExperiment(
        model_initializer=XGBClassifier, model_init_params=dict(subsample=0.01)
    )
    # Pro Tip: By setting XGBoost's subsample ridiculously low, we can get bad scores on purpose

    # Upon completion of this Experiment, we see a warning that not all result files will be saved
    # This is because the final score of the Experiment was below our threshold of 0.75
    # Specifically, we skipped saving prediction files (OOF, holdout, test, or in-fold), and the heartbeat file

    # What still got saved is the Experiment's: key information, leaderboard position, and description file
    # These are saved to allow us to use the information for future hyperparameter optimization, and detect repeated Experiments
    # Additionally, the Experiment's script backup is saved, but that's because its one of the first things that happens
    # For even finer control over what gets saved, use `do_full_save` together with `file_blacklist`

    # Now, lets perform another Experiment that does a bit better than our intentionally miserable one
    experiment_1 = CrossValidationExperiment(
        model_initializer=XGBClassifier, model_init_params=dict(subsample=0.5)
    )
    # Our second Experiment was executed in the same Environment, so it was still subject to the `do_full_save` constraint
    # However, because it scored above 0.75 (hopefully), all of the result files were saved


if __name__ == "__main__":
    execute()
