from hyperparameter_hunter import Environment, CrossValidationExperiment
from hyperparameter_hunter import GBRT, Real, Integer, Categorical
from hyperparameter_hunter.utils.learning_utils import get_diabetes_data
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


def execute():
    env = Environment(
        train_dataset=get_diabetes_data(target="target"),
        root_results_path="HyperparameterHunterAssets",
        metrics_map=dict(r2_score=lambda t, p: -r2_score(t, p)),
        cross_validation_type="KFold",
        cross_validation_params=dict(n_splits=10, shuffle=True, random_state=32),
    )

    #################### Experiment ####################
    experiment = CrossValidationExperiment(
        model_initializer=XGBRegressor,
        model_init_params=dict(
            objective="reg:linear", max_depth=4, n_estimators=400, subsample=0.5
        ),
        model_extra_params=dict(fit=dict(eval_metric="mae")),
    )

    #################### Optimization ####################
    optimizer = GBRT(iterations=20, read_experiments=True, random_state=32)

    optimizer.set_experiment_guidelines(
        model_initializer=XGBRegressor,
        model_init_params=dict(
            max_depth=Integer(2, 20),
            learning_rate=Real(0.01, 0.7),
            n_estimators=Integer(100, 500),
            subsample=0.5,
            booster=Categorical(["gbtree", "gblinear"]),
        ),
        model_extra_params=dict(fit=dict(eval_metric=Categorical(["rmse", "mae"]))),
    )

    optimizer.go()


if __name__ == "__main__":
    execute()
