##################################################
# Make Executable in Terminal, and Enable Module Importing
##################################################
import sys
import os.path

try:
    sys.path.append(os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0])
except Exception as _ex:
    raise _ex

##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.importer import hook_keras_layer
try:
    hook_keras_layer()
except Exception as _ex:
    raise

from hyperparameter_hunter.environment import Environment
from hyperparameter_hunter.experiments import CrossValidationExperiment
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data

##################################################
# Import Miscellaneous Assets
##################################################
import copy

##################################################
# Import Learning Assets
##################################################
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Activation, Dropout
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor


# def get_holdout_set(train, target_column):
#     return train, train.copy()


def define_architecture(input_dim=-1):
    model = Sequential([
        Dense(100, kernel_initializer='uniform', input_dim=input_dim, activation='relu'),
        # Dropout(0.4),
        Dropout(0.5),

        Dense(50, kernel_initializer='uniform', activation='relu'),
        Dropout(0.3),

        Dense(1, kernel_initializer='uniform', activation='sigmoid'),
    ])

    model.compile(
        optimizer=Adam(),
        loss=binary_crossentropy,
        metrics=['accuracy'],
    )

    return model


def execute():
    env = Environment(
        train_dataset=get_breast_cancer_data(),
        root_results_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../HyperparameterHunterAssets')),
        target_column='diagnosis',
        metrics_map=['roc_auc_score'],
        runs=2,
        cross_validation_type=StratifiedKFold,
        cross_validation_params=dict(n_splits=5, shuffle=True, random_state=32),
    )

    experiment = CrossValidationExperiment(
        model_initializer=KerasClassifier,
        model_init_params=dict(build_fn=define_architecture),
        # model_init_params=define_architecture,
        model_extra_params=dict(
            callbacks=[
                ModelCheckpoint(filepath='../foo_checkpoint'),
                ReduceLROnPlateau(patience=5),
            ],
            batch_size=32,
            epochs=10,
            verbose=2,
            shuffle=True,
        ),
    )


if __name__ == '__main__':
    execute()
