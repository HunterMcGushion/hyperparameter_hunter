from hyperparameter_hunter import Environment, CVExperiment, BayesianOptPro
from hyperparameter_hunter import Real, Integer, Categorical
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
import os.path
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier


def _build_fn_experiment(input_shape):
    model = Sequential(
        [
            Dense(100, kernel_initializer="uniform", input_shape=input_shape, activation="relu"),
            Dropout(0.5),
            Dense(1, kernel_initializer="uniform", activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_fn_optimization(input_shape):
    model = Sequential(
        [
            Dense(
                Integer(50, 150),
                kernel_initializer="uniform",
                input_shape=input_shape,
                activation="relu",
            ),
            Dropout(Real(0.2, 0.7)),
            Dense(1, kernel_initializer="uniform", activation=Categorical(["sigmoid", "relu"])),
        ]
    )
    model.compile(
        optimizer=Categorical(["adam", "rmsprop"]), loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def _execute():
    #################### Environment ####################
    env = Environment(
        train_dataset=get_breast_cancer_data(target="target"),
        results_path="HyperparameterHunterAssets",
        metrics=["roc_auc_score"],
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=5, shuffle=True, random_state=32),
    )

    #################### Experimentation ####################
    experiment = CVExperiment(
        model_initializer=KerasClassifier,
        model_init_params=dict(build_fn=_build_fn_experiment),
        model_extra_params=dict(
            callbacks=[ReduceLROnPlateau(patience=5)], batch_size=32, epochs=10, verbose=0
        ),
    )

    #################### Optimization ####################
    optimizer = BayesianOptPro(iterations=10)
    optimizer.set_experiment_guidelines(
        model_initializer=KerasClassifier,
        model_init_params=dict(build_fn=_build_fn_optimization),
        model_extra_params=dict(
            callbacks=[ReduceLROnPlateau(patience=Integer(5, 10))],
            batch_size=Categorical([32, 64], transform="onehot"),
            epochs=10,
            verbose=0,
        ),
    )
    optimizer.go()


if __name__ == "__main__":
    _execute()
