"""This example demonstrates how to optimize `class_weight` values, but may be applied to other
hyperparameters that are inside some nested object. Although this example uses SKLearn's
`RandomForestClassifier`, similar `class_weight` kwargs in other libraries can be optimized in the
same way, such as LightGBM's `LGBMClassifier` `class_weight` kwarg"""
from hyperparameter_hunter import Environment, CVExperiment, BayesianOptPro, Integer, Categorical
from hyperparameter_hunter.utils.learning_utils import get_iris_data
from sklearn.ensemble import RandomForestClassifier


def execute():
    #################### Environment ####################
    env = Environment(
        train_dataset=get_iris_data(),
        results_path="HyperparameterHunterAssets",
        target_column="species",
        metrics=["hamming_loss"],
        cv_params=dict(n_splits=5, random_state=32),
    )

    #################### Experiment ####################
    # Just a reference for normal `class_weight` usage outside of optimization
    CVExperiment(RandomForestClassifier, dict(n_estimators=10, class_weight={0: 1, 1: 1, 2: 1}))

    #################### Optimization ####################
    opt = BayesianOptPro(iterations=10, random_state=32)
    opt.forge_experiment(
        model_initializer=RandomForestClassifier,
        model_init_params=dict(
            # Weight values for each class can be optimized with `Categorical`/`Integer`
            class_weight={
                0: Categorical([1, 3]),
                1: Categorical([1, 4]),
                2: Integer(1, 9),  # You can also use `Integer` for low/high ranges
            },
            criterion=Categorical(["gini", "entropy"]),
            n_estimators=Integer(5, 100),
        ),
    )
    opt.go()


if __name__ == "__main__":
    execute()
