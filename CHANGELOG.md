<a name="1.0.0"></a>
### 1.0.0 (2018-08-19)

#### Features
* Simplified providing hyperparameter search dimensions during optimization
    * Old method of providing search dimensions:

        ```python
        from hyperparameter_hunter import BayesianOptimization, Real, Integer, Categorical

        optimizer = BayesianOptimization(
            iterations=100, read_experiments=True, dimensions=[
                Integer(name='max_depth', low=2, high=20),
                Real(name='learning_rate', low=0.0001, high=0.5),
                Categorical(name='booster', categories=['gbtree', 'gblinear', 'dart'])
            ]
        )
        optimizer.set_experiment_guidelines(
            model_initializer=XGBClassifier,
            model_init_params=dict(n_estimators=200, subsample=0.5, learning_rate=0.1)
        )
        optimizer.go()
        ```
    * New method:

        ```python
        from hyperparameter_hunter import BayesianOptimization, Real, Integer, Categorical

        optimizer = BayesianOptimization(iterations=100, read_experiments=True)
        optimizer.set_experiment_guidelines(
            model_initializer=XGBClassifier,
            model_init_params=dict(
                n_estimators=200, subsample=0.5,
                learning_rate=Real(0.0001, 0.5),
                max_depth=Integer(2, 20),
                booster=Categorical(['gbtree', 'gblinear', 'dart'])
            )
        )
        optimizer.go()
        ```
    * The `dimensions` kwarg is removed from the OptimizationProtocol classes, and hyperparameter search dimensions are now provided along with the concrete hyperarameters via `set_experiment_guidelines`. If a value is a descendant of `hyperparameter_hunter.space.Dimension`, it is automatically detected as a space to be searched and optimized
* Improved support for Keras hyperparameter optimization
    * Keras Experiment:

        ```python
        from hyperparameter_hunter import CrossValidationExperiment
        from keras import *

        def build_fn(input_shape):
            model = Sequential([
                Dense(100, kernel_initializer='uniform', input_shape=input_shape, activation='relu'),
                Dropout(0.5),
                Dense(1, kernel_initializer='uniform', activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

        experiment = CrossValidationExperiment(
            model_initializer=KerasClassifier,
            model_init_params=build_fn,
            model_extra_params=dict(
                callbacks=[ReduceLROnPlateau(patience=5)],
                batch_size=32, epochs=10, verbose=0
            )
        )
        ```
    * Keras Optimization:

        ```python
        from hyperparameter_hunter import Real, Integer, Categorical, RandomForestOptimization
        from keras import *

        def build_fn(input_shape):
            model = Sequential([
                Dense(Integer(50, 150), input_shape=input_shape, activation='relu'),
                Dropout(Real(0.2, 0.7)),
                Dense(1, activation=Categorical(['sigmoid', 'softmax']))
            ])
            model.compile(
                optimizer=Categorical(['adam', 'rmsprop', 'sgd', 'adadelta']),
                loss='binary_crossentropy', metrics=['accuracy']
            )
            return model

        optimizer = RandomForestOptimization(iterations=7)
        optimizer.set_experiment_guidelines(
            model_initializer=KerasClassifier,
            model_init_params=build_fn,
            model_extra_params=dict(
                callbacks=[ReduceLROnPlateau(patience=Integer(5, 10))],
                batch_size=Categorical([32, 64]),
                epochs=10, verbose=0
            )
        )
        optimizer.go()
        ```
* Lots of other new features and bug-fixes

<a name="0.0.1"></a>
### 0.0.1 (2018-06-14)

#### Features
* Initial release