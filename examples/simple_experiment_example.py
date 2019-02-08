from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter.utils.learning_utils import get_toy_classification_data
from xgboost import XGBClassifier


def execute():
    env = Environment(
        train_dataset=get_toy_classification_data(),
        results_path="HyperparameterHunterAssets",
        metrics_map=["roc_auc_score"],
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=5, shuffle=True, random_state=32),
    )

    experiment = CVExperiment(
        model_initializer=XGBClassifier,
        model_init_params=dict(
            objective="reg:linear", max_depth=3, n_estimators=100, subsample=0.5
        ),
    )


if __name__ == "__main__":
    execute()
