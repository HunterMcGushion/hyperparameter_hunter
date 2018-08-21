from hyperparameter_hunter import Environment, CrossValidationExperiment, Real, Integer, Categorical
from hyperparameter_hunter import BayesianOptimization, DummySearch, ExtraTreesOptimization, RandomForestOptimization
from hyperparameter_hunter.utils.learning_utils import get_breast_cancer_data
import os.path
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier


def experiment_0_builder(input_shape):
# def experiment_0_builder(input_dim):
    """Simple Multilayer Perceptron architecture - nothing fancy

    Notes
    -----
    Dense layers are given `kernel_initializer` values of 'glorot_uniform'. This happens to be the default value for this kwarg,
    so in this case, including it is superfluous, but we'll see why its interesting in a minute

    Format of this function is essentially whatever you would normally do for the `build_fn` expected when using the
    `keras.wrappers.scikit_learn` wrappers (`KerasClassifier`, `KerasRegressor`), with the following difference

    To specify the `input_shape` in your first layer, we have two options: 1) provide the `input_shape` argument, which is
    automatically calculated and passed through by `hyperparameter_hunter`; or 2) manually provide an `input_shape` tuple.
    # TODO: REVISE BELOW IF SIGNATURE MODIFICATION DOES NOT GET ADDED TO EXPERIMENTS, LIKE OPTIMIZATION PROTOCOLS
    # TODO: REVISE BELOW IF SIGNATURE MODIFICATION DOES NOT GET ADDED TO EXPERIMENTS, LIKE OPTIMIZATION PROTOCOLS
    Also
    note that when defining your own `build_fn` (like this one), you don't even need to actually include an `input_shape` argument
    in the function's signature to be able to use it in the first layer of your model. `hyperparameter_hunter` will check the
    signature for you, and modify it if necessary before it gets started. See :func:`experiment_1_builder` for an example"""
    # TODO: REVISE ABOVE IF SIGNATURE MODIFICATION DOES NOT GET ADDED TO EXPERIMENTS, LIKE OPTIMIZATION PROTOCOLS
    # TODO: REVISE ABOVE IF SIGNATURE MODIFICATION DOES NOT GET ADDED TO EXPERIMENTS, LIKE OPTIMIZATION PROTOCOLS
    model = Sequential([
        Dense(100, kernel_initializer='glorot_uniform', input_shape=input_shape, activation='relu'),
        # Dense(100, kernel_initializer='glorot_uniform', input_dim=input_dim, activation='relu'),
        Dropout(0.5),
        Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid')
    ])
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],
    )
    return model


def experiment_1_builder(input_shape):
# def experiment_1_builder(input_dim):
    """This is different from :func:`experiment_0_builder` in two ways: 1) it uses the `input_shape` argument in the first Dense
    layer, despite not having explicitly included it in the function signature; and 2) the `kernel_initializer` arguments to both
    Dense layers have been removed. We expect this to do exactly the same thing as :func:`experiment_0_builder` assuming their
    extra parameters given below are also identical"""
    model = Sequential([
        Dense(100, input_shape=input_shape, activation='relu'),
        # Dense(100, input_dim=input_dim, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],
    )
    return model


def experiment_2_builder(input_shape):
    pass


def experiment_3_builder(input_shape):
    pass


def experiment_4_builder(input_shape):
    pass


def optimization_0_builder(input_shape):
    pass


def optimization_1_builder(input_shape):
    pass


def optimization_2_builder(input_shape):
    pass


# noinspection PyTypeChecker
def _execute():
    env = Environment(
        train_dataset=get_breast_cancer_data(target='target'),
        root_results_path='./ExtendedKerasAssets',
        metrics_map=['roc_auc_score'],
        cross_validation_type='StratifiedKFold',
        cross_validation_params=dict(n_splits=5, shuffle=True, random_state=32)
    )

    extra_params_0 = dict(
        callbacks=[ReduceLROnPlateau(patience=5)],
        batch_size=32,
        epochs=10
    )

    exp_0 = CrossValidationExperiment(KerasClassifier, experiment_0_builder, model_extra_params=extra_params_0)

    exp_1 = CrossValidationExperiment(KerasClassifier, experiment_1_builder, model_extra_params=extra_params_0)

    print('0:  ID={}   CE_KEY={}   H_KEY={}'.format(exp_0.experiment_id, exp_0.cross_experiment_key, exp_0.hyperparameter_key))
    print('1:  ID={}   CE_KEY={}   H_KEY={}'.format(exp_1.experiment_id, exp_1.cross_experiment_key, exp_1.hyperparameter_key))

    print()


if __name__ == '__main__':
    _execute()
