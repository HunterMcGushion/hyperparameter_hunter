from hyperparameter_hunter import Environment, CVExperiment
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import load_digits


def prep_data(n_class=10):
    input_data, target_data = load_digits(n_class=n_class, return_X_y=True)
    train_df = pd.DataFrame(
        data=input_data, columns=["c_{:02d}".format(_) for _ in range(input_data.shape[1])]
    )

    train_df["target"] = target_data
    train_df = pd.get_dummies(train_df, columns=["target"], prefix="target")
    return train_df


def build_fn(input_shape=-1):
    model = Sequential(
        [
            Reshape((8, 8, -1), input_shape=(64,)),
            Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _execute():
    env = Environment(
        train_dataset=prep_data(),
        results_path="HyperparameterHunterAssets",
        metrics=["roc_auc_score"],
        target_column=[f"target_{_}" for _ in range(10)],
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=10, shuffle=True, random_state=True),
    )

    experiment = CVExperiment(
        model_initializer=KerasClassifier,
        model_init_params=build_fn,
        model_extra_params=dict(batch_size=32, epochs=10, verbose=0, shuffle=True),
    )


if __name__ == "__main__":
    _execute()
