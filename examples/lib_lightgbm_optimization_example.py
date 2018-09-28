from hyperparameter_hunter import Environment, Real, Integer, Categorical, RandomForestOptimization
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier


def _execute():
    env = Environment(
        train_dataset=get_breast_cancer_data(),
        root_results_path="HyperparameterHunterAssets",
        target_column="diagnosis",
        metrics_map=["roc_auc_score"],
        cross_validation_type=StratifiedKFold,
        cross_validation_params=dict(n_splits=10, shuffle=True, random_state=32),
        runs=1,
    )

    optimizer = RandomForestOptimization(iterations=10, read_experiments=True)
    optimizer.set_experiment_guidelines(
        model_initializer=LGBMClassifier,
        model_init_params=dict(
            boosting_type=Categorical(["gbdt", "dart"]),
            num_leaves=Integer(5, 20),
            max_depth=-1,
            min_child_samples=5,
            subsample=0.5,
        ),
    )
    optimizer.go()


if __name__ == "__main__":
    _execute()
