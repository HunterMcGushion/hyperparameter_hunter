from hyperparameter_hunter import Environment, Real, Integer, Categorical, BayesianOptimization
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def _execute():
    env = Environment(
        train_dataset=get_breast_cancer_data(),
        results_path="HyperparameterHunterAssets",
        target_column="diagnosis",
        metrics=["roc_auc_score"],
        cv_type=StratifiedKFold,
        cv_params=dict(n_splits=10, shuffle=True, random_state=32),
        runs=2,
    )

    optimizer = BayesianOptimization(iterations=10, read_experiments=True, random_state=None)

    optimizer.set_experiment_guidelines(
        model_initializer=XGBClassifier,
        model_init_params=dict(
            max_depth=Integer(2, 20),
            learning_rate=Real(0.0001, 0.5),
            n_estimators=200,
            subsample=0.5,
            booster=Categorical(["gbtree", "gblinear", "dart"]),
        ),
        model_extra_params=dict(fit=dict(eval_metric=Categorical(["auc", "rmse", "mae"]))),
    )

    optimizer.go()


if __name__ == "__main__":
    _execute()
