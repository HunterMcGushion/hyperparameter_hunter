from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
import os.path
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier


def build_fn(input_shape=-1):
    model = Sequential(
        [
            Dense(100, kernel_initializer="uniform", input_shape=input_shape, activation="relu"),
            Dropout(0.5),
            Dense(50, kernel_initializer="uniform", activation="relu"),
            Dropout(0.3),
            Dense(1, kernel_initializer="uniform", activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def execute():
    env = Environment(
        train_dataset=get_breast_cancer_data(),
        results_path="HyperparameterHunterAssets",
        target_column="diagnosis",
        metrics_map=["roc_auc_score"],
        cross_validation_type="StratifiedKFold",
        cross_validation_params=dict(n_splits=5, shuffle=True, random_state=32),
    )

    experiment = CVExperiment(
        model_initializer=KerasClassifier,
        model_init_params=build_fn,
        model_extra_params=dict(
            callbacks=[
                ModelCheckpoint(
                    filepath=os.path.abspath("foo_checkpoint"), save_best_only=True, verbose=1
                ),
                ReduceLROnPlateau(patience=5),
            ],
            batch_size=32,
            epochs=10,
            verbose=0,
            shuffle=True,
        ),
    )


if __name__ == "__main__":
    execute()
