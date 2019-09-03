from hyperparameter_hunter import Environment, CVExperiment, FeatureEngineer
from hyperparameter_hunter import BayesianOptPro, Real, Integer, Categorical

import os

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
except Exception:
    raise

import pandas as pd
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier


##################################################
# Format MNIST DataFrames
##################################################
def prep_data():
    (train_input, train_target), (holdout_input, holdout_target) = mnist.load_data()

    #################### Flatten Input Data ####################
    num_flat_cols = train_input.shape[1] * train_input.shape[2]  # 784 == (28 * 28)
    train_input = train_input.reshape(train_input.shape[0], num_flat_cols)
    holdout_input = holdout_input.reshape(holdout_input.shape[0], num_flat_cols)

    #################### Train DataFrame ####################
    train_df = pd.DataFrame(data=train_input)
    train_df["target"] = train_target
    train_df = pd.get_dummies(train_df, columns=["target"], prefix="target")

    #################### Holdout DataFrame ####################
    holdout_df = pd.DataFrame(data=holdout_input)
    holdout_df["target"] = holdout_target
    holdout_df = pd.get_dummies(holdout_df, columns=["target"], prefix="target")

    return train_df, holdout_df


##################################################
# Experiment Model Builder
##################################################
def build_fn_exp(input_shape=-1):
    model = Sequential(
        [
            Reshape((28, 28, -1), input_shape=(784,)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


##################################################
# Optimization Model Builder
##################################################
def build_fn_opt(input_shape=-1):
    model = Sequential(
        [
            Reshape((28, 28, -1), input_shape=(784,)),
            Conv2D(
                Categorical([32, 64, 128]),
                kernel_size=Categorical([(2, 2), (3, 3), (4, 4)]),
                activation="relu",
            ),
            Conv2D(
                Categorical([32, 64, 128]),
                kernel_size=Categorical([(2, 2), (3, 3), (4, 4)]),
                activation="relu",
            ),
            MaxPooling2D(pool_size=Categorical([(2, 2), (3, 3)])),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


##################################################
# Go!
##################################################
def execute():
    train_df, holdout_df = prep_data()

    env = Environment(
        train_dataset=train_df,
        results_path="HyperparameterHunterAssets",
        metrics=["roc_auc_score"],
        target_column=[f"target_{_}" for _ in range(10)],  # 10 classes (one-hot-encoded output)
        holdout_dataset=holdout_df,
        cv_type="StratifiedKFold",
        cv_params=dict(n_splits=3, shuffle=True, random_state=True),
    )

    exp = CVExperiment(KerasClassifier, build_fn_exp, dict(batch_size=64, epochs=10, verbose=1))

    opt = BayesianOptPro(iterations=10, random_state=32)
    opt.forge_experiment(KerasClassifier, build_fn_opt, dict(batch_size=64, epochs=10, verbose=0))
    opt.go()


if __name__ == "__main__":
    execute()
