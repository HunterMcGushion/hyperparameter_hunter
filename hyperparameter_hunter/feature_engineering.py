"""This module is still in an experimental stage and should not be assumed to be "reliable", or
"useful", or anything else that might be expected of a normal module"""
##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.utils.general_utils import type_val
from hyperparameter_hunter.utils.learning_utils import upsample
from hyperparameter_hunter.settings import G

##################################################
# Import Miscellaneous Assets
##################################################
from collections import Counter

##################################################
# Import Learning Assets
##################################################
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler


class PreprocessingPipelineMixIn(object):
    def __init__(
        self,
        pipeline,
        preprocessing_params,
        features,
        target_column,
        train_input_data: object = None,
        train_target_data: object = None,
        validation_input_data: object = None,
        validation_target_data: object = None,
        holdout_input_data: object = None,
        holdout_target_data: object = None,
        test_input_data: object = None,
        fitting_guide=None,
        fail_gracefully=False,
        preprocessing_stage="infer",
    ):
        """

        Parameters
        ----------
        pipeline: List
            List of tuples of form: (<string id>, <callable function>), in which the id identifies
            the paired function transformations to be fitted on dfs specified by fitting_guide[i],
            then applied to all other same-type, non-null dfs
        preprocessing_params: Dict
            All the parameters necessary for the desired preprocessing functionality
        features: List
            List containing strings that specify the columns to be used as input
        target_column: String
            String naming the target column
        train_input_data: Pandas Dataframe
            ...
        train_target_data: Pandas Dataframe
            ...
        validation_input_data: Pandas Dataframe
            ...
        validation_target_data: Pandas Dataframe
            ...
        holdout_input_data: Pandas Dataframe
            ...
        holdout_target_data: Pandas Dataframe
            ...
        test_input_data: Pandas Dataframe
            ...
        fitting_guide: List of same length as pipeline containing tuples of strings, default=None
            If not None, specifies datasets used to fit each manipulation in pipeline. Those not
            included in the list (and of same type: input/target) will be transformed according to
            the fitted functions. Else, infer from preprocessing_stage
        fail_gracefully: Boolean, default=False
            If True, Exceptions thrown by preprocessing transformations will be logged and skipped,
            so processing can continue
        preprocessing_stage: String in ['pre_cv', 'intra_cv', 'infer'], default='infer'
            Denotes when preprocessing is occurring. If 'pre_cv', pipeline functions are fit on all
            available data. If 'intra_cv', pipeline functions are fit on train data and applied to
            all same-type, non-null data. Else, infer stage"""
        ##################################################
        # Core Attributes
        ##################################################
        self.pipeline = pipeline
        self.preprocessing_params = preprocessing_params
        self.features = features
        self.target_column = target_column

        ##################################################
        # Dataset Attributes
        ##################################################
        self.train_input_data = train_input_data.copy() if train_input_data is not None else None
        self.train_target_data = train_target_data.copy() if train_target_data is not None else None
        self.validation_input_data = (
            validation_input_data.copy() if validation_input_data is not None else None
        )
        self.validation_target_data = (
            validation_target_data.copy() if validation_target_data is not None else None
        )
        self.holdout_input_data = (
            holdout_input_data.copy() if holdout_input_data is not None else None
        )
        self.holdout_target_data = (
            holdout_target_data.copy() if holdout_target_data is not None else None
        )
        self.test_input_data = test_input_data.copy() if test_input_data is not None else None

        ##################################################
        # Miscellaneous Attributes
        ##################################################
        self.fitting_guide = fitting_guide
        self.fail_gracefully = fail_gracefully

        ##################################################
        # Preprocessing Stage and Dataset Type Attributes
        ##################################################
        self.preprocessing_stage = preprocessing_stage
        self.all_input_sets = self.get_non_null(
            ["{}_input_data".format(_) for _ in ("train", "validation", "holdout", "test")]
        )
        self.all_target_sets = self.get_non_null(
            ["{}_target_data".format(_) for _ in ("train", "validation", "holdout")]
        )
        self.fit_input_sets = None
        self.fit_target_sets = None

        ##################################################
        # Initialize Mix-Ins/Inherited Classes
        ##################################################
        pass

        ##################################################
        # Ensure Attributes are Properly Initialized
        ##################################################
        self.set_preprocessing_stage_and_sets()

    def get_non_null(self, dataset_names):
        return [_ for _ in dataset_names if self.__getattribute__(_) is not None]

    def set_preprocessing_stage_and_sets(self):
        """Ensures preprocessing_stage has been properly initialized before initializing
        fit_input_sets and fit_target_sets"""
        try:
            self.preprocessing_stage = self.initialize_preprocessing_stage()
        except Exception as _ex:
            raise (_ex)
        else:
            if self.preprocessing_stage == "pre_cv":
                self.fit_input_sets = self.all_input_sets
                self.fit_target_sets = self.all_target_sets
                # self.fit_target_sets = ['train_target_data', 'holdout_target_data']
            elif self.preprocessing_stage == "intra_cv":
                self.fit_input_sets = ["train_input_data"]
                self.fit_target_sets = ["train_target_data"]
                # self.fit_input_sets = ['train_input_data', 'validation_input_data', 'holdout_input_data', 'test_input_data']
                # self.fit_target_sets = ['train_target_data', 'validation_target_data', 'holdout_target_data']

    def initialize_preprocessing_stage(self):
        """Ensures preprocessing_stage can be set according to class attributes or method input"""
        _stages, _err = ["pre_cv", "intra_cv"], "Unknown error occurred."
        _i_strs = ["validation_input_data", "validation_target_data"]
        _i_sets = [getattr(self, _) for _ in _i_strs]

        if self.preprocessing_stage in _stages:
            return self.preprocessing_stage
        elif self.preprocessing_stage == "infer":
            if all([_ for _ in _i_sets]):
                return "intra_cv"
            elif any([_ for _ in _i_sets]):
                _err = "Inference failed. {} types must be same. Received: {}".format(
                    _i_strs, [type(_) for _ in _i_sets]
                )
            else:
                return "pre_cv"
        else:
            _err = "preprocessing_stage must be in {}. Received type {}: {}".format(
                _stages, *type_val(self.preprocessing_stage)
            )

        if self.fail_gracefully is True:
            G.warn(_err)
            return "pre_cv"
        else:
            raise ValueError(_err)

    def build_pipeline(self):
        new_pipeline = []  # (<str id>, <callable transformation>, <sets to fit on>, <sets to transform>)

        if not isinstance(self.pipeline, list):
            raise TypeError(
                "Expected pipeline of type list. Received {}: {}".format(*type_val(self.pipeline))
            )

        for i, step in enumerate(self.pipeline):
            step_id, step_callable, step_fit_sets, step_transform_sets = None, None, None, None

            ##################################################
            # Pipeline is a list of strings
            ##################################################
            if isinstance(step, str):
                # element names a method in this class to use
                step_id, step_callable = step, getattr(self, step, default=None)
                if step_callable is None:
                    raise AttributeError(
                        "Expected pipeline value to name a method. Received {}: {}".format(
                            *type_val(step)
                        )
                    )
            ##################################################
            # Pipeline is a list of tuple/list pairs
            ##################################################
            # TODO: Instead of forcing len() == 2, merge self.fitting_guide into self.pipeline.
            # TODO: Max valid length == 4 after adding fit_sets(subset of transform_sets), transform_sets
            # TODO: If len > 4, throw WARNING that extra values will be ignored and continue
            elif any([isinstance(step, _) for _ in (tuple, list)]) and len(step) == 2:
                # element is a tuple/list of length 2, where 2nd value is a callable or names method of transformation
                if isinstance(step[0], str):
                    step_id = step[0]
                else:
                    pass

                if callable(step[1]):
                    pass
                    # TODO: Dynamically create new method, whose name is heavily mangled and modified
                    # TODO: Try to include original callable __name__ in new method's name. If unavailable use "i"
                    # TODO: New method name should be something like: "__dynamic_pipeline_method_{}".format(step[1].__name__ or i)
                    # TODO: Actual name that would be called would be mangled because of double underscore prefix
                    # FLAG: If you want your method to have other arguments, place them in preprocessing_params dict, instead
                    # FLAG: Then just use self.preprocessing_params[<your arg>] inside your callable
                    # FLAG: In fact, declaring values that may change later on directly in your callable could be problematic
                    # FLAG: If you include them in preprocessing_params, hyperparameters for experiment will be clear
            ##################################################
            # Pipeline type is invalid
            ##################################################
            else:
                raise TypeError(
                    "Expected pipeline step to be: a str, or a tuple pair. Received {}: {}".format(
                        *type_val(step)
                    )
                )

            ##################################################
            # Additional Error Handling
            ##################################################
            if step_id is None:
                raise TypeError(
                    "Expected str as first value in each pipeline tuple. Received {}: {}".format(
                        *type_val(step[0])
                    )
                )

            new_pipeline.append((step_id, step_callable, step_fit_sets, step_transform_sets))

    def custom_pipeline_method_builder(self, functionality, name=None):
        """...

        Parameters
        ----------
        functionality: Callable
            Performs all desired transformations/alterations/work for this pipeline step. This
            callable will not receive any input arguments, so don't expect any. Instead, it is
            implemented as a class method, so it has access to all class attributes and methods. To
            work properly, the class attributes: ['self.train_input_data', 'self.train_target_data',
            'self.validation_input_data', 'self.validation_target_data', 'self.holdout_input_data',
            'self.holdout_target_data', 'self.test_input_data'] are expected to be directly
            modified. See the "Notes"/"Examples" sections below for more
        name: String, or None, default=None
            Suffix for the name of the new custom method. See below "Notes" section for details on
            method name creation

        Returns
        -------
        name: str
            The name of the new method that was created

        Notes
        -----
        WARNING: Because the custom functionality is implemented as a class method, it is capable
        of modifying values that are not expected to change, or setting new attributes. Doing either
        of these is a bad idea. The only attributes that should be set are those listed in the above
        "Parameters" description for the "functionality" argument. Additionally, the only values
        that should be retrieved are the aforementioned "data" attributes, plus
        :attr:`preprocessing_params`

        METHOD ARGUMENTS: If the custom functionality requires some input argument that could be
        subject to change later (like a hyperparameter), it should be included in
        :attr:`preprocessing_params`. Then in the custom functionality, it can be retrieved with
        "self.preprocessing_params[<your_arg>]". See the "Examples" section below for details on how
        to do this. The two primary reasons for this behavior are as follows:

        1) to get around having to make sense of methods' expected arguments and the arguments
        actually input to them, and
        2) to include any necessary arguments in the experiment's hyperparameters.

        Examples
        --------
        >>> from hyperparameter_hunter.feature_engineering import PreprocessingPipelineMixIn
        >>> def my_function(self):
        >>>     self.train_input_data = self.train_input_data.fillna(self.preprocessing_params['my_imputer'])
        Notice in "my_function", "self" is the only input, "self.train_input_data" is directly
        modified, and instead of passing "my_imputer" as an input, it is referenced in
        "self.preprocessing_params". Now, the class can use "my_function" below.
        >>> preprocessor = PreprocessingPipelineMixIn(
        >>>     pipeline=[('my_function', my_function)],
        >>>     preprocessing_params=dict(my_imputer=-1), features=[], target_column=''
        >>> )
        The "pipeline" is set to include "my_function", which, after its creation, will be able to
        retrieve "my_imputer" from "self.preprocessing_params". Note that this example just
        demonstrates custom method building. It won't work as-is, without any train_input_data,
        among other things. Now in a later experiment, null values can be imputed to -2 instead of
        -1, just by changing "preprocessing_params":
        >>> preprocessor = PreprocessingPipelineMixIn(
        >>>     pipeline=[('my_function', my_function)],
        >>>     preprocessing_params=dict(my_imputer=-2), features=[], target_column=''
        >>> )
        This makes it much easier to keep track of the actual hyperparameters being used in an
        experiment than having to scour obscure functions for some number that may or may not even
        be declared inside"""
        if not callable(functionality):
            raise TypeError(
                "Custom pipeline methods must be callable. Received type {}".format(
                    type(functionality)
                )
            )

        # TODO: Set name (using "functionality.__name__") if name is None

        while hasattr(self, name):
            _name = name + ""  # TODO: Make changes to "name" here
            # TODO: Do something to further modify name and check again
            G.warn(
                'Encountered naming conflict in custom_pipeline_method_builder with "{}". Trying "{}"'.format(
                    name, _name
                )
            )
            name = _name

        #################### Create New Custom Method ####################
        setattr(self, name, functionality)

        return name

    def data_imputation(self, which_sets=None):
        imputer = self.preprocessing_params.get("imputer", None)
        which_sets = which_sets if which_sets else self.fit_input_sets

        for data_key in which_sets:
            data = self.__getattribute__(data_key)

            if data is not None:
                if callable(imputer):  # Apply Function to Impute Data
                    # TODO: Send either "self" or all attributes in self as other input to "imputer"
                    # TODO: Force callable "imputer" to have **kwargs, or check for the args it expects and send only those
                    self.__setattr__(data_key, imputer(data))
                elif any(
                    [isinstance(imputer, _) for _ in (int, float)]
                ):  # Fill Null Data With Given Value
                    self.__setattr__(data_key, data.fillna(imputer))

        G.log("Completed data_imputation preprocessing")

    def target_data_transformation(self, which_sets=None):
        transformation = self.preprocessing_params.get("target_transformation", None)
        which_sets = which_sets if which_sets else self.fit_target_sets

        for data_key in which_sets:
            data = self.__getattribute__(data_key)

            if callable(transformation) and data:
                # TODO: Send either "self" or all attributes in self as other input to "imputer"
                # TODO: Force callable "imputer" to have **kwargs, or check for the args it expects and send only those
                self.__setattr__(data_key, transformation(data))

        G.log("Completed target_data_transformation preprocessing")

    def data_scaling(self, which_sets=None):
        which_sets = which_sets if which_sets else self.fit_input_sets

        # TODO: Expand method to include other scaling types by sending string param or callable for apply_scale arg
        if self.preprocessing_params.get("apply_standard_scale", False) is True:
            scaler = StandardScaler()

            # TODO: Modify fitting process to use 'which_sets' and 'self.fit_input_sets' like 'data_imputation' method
            scaler.fit(self.train_input_data[self.features].values)

            if "train_input_data" in self.all_input_sets:
                self.train_input_data[self.features] = scaler.transform(
                    self.train_input_data[self.features].values
                )
            if "holdout_input_data" in self.all_input_sets:
                self.holdout_input_data[self.features] = scaler.transform(
                    self.holdout_input_data[self.features].values
                )
            if "test_input_data" in self.all_input_sets:
                self.test_input_data[self.features] = scaler.transform(
                    self.test_input_data[self.features].values
                )

        G.log(
            'Completed data_scaling preprocessing. preprocessing_params["apply_standard_scale"]={}'.format(
                self.preprocessing_params.get("apply_standard_scale", False)
            )
        )


class PreCVPreprocessingPipeline(PreprocessingPipelineMixIn):
    def __init__(
        self,
        features,
        target_column,
        train_input_data,
        train_target_data,
        holdout_input_data,
        holdout_target_data,
        test_input_data=None,
    ):
        PreprocessingPipelineMixIn.__init__(
            self,
            preprocessing_stage="pre_cv",
            pipeline=None,
            fitting_guide=None,
            preprocessing_params=None,
            features=features,
            target_column=target_column,
            train_input_data=train_input_data,
            train_target_data=train_target_data,
            validation_input_data=None,
            validation_target_data=None,
            holdout_input_data=holdout_input_data,
            holdout_target_data=holdout_target_data,
            test_input_data=test_input_data,
            fail_gracefully=False,
        )

    # FLAG: WARNING: Method of same name in "CrossValidationWrapper" class
    # FLAG: WARNING: Method of same name in "CrossValidationWrapper" class
    def pre_cv_preprocessing(self):
        # FLAG: WARNING: Method of same name in "CrossValidationWrapper" class
        # FLAG: WARNING: Method of same name in "CrossValidationWrapper" class
        #################### Feature Selection ####################
        pass

        #################### Impute Missing Values in Data ####################
        pass


class IntraCVPreprocessingPipeline(PreprocessingPipelineMixIn):
    def __init__(
        self,
        features,
        target_column,
        train_input_data,
        train_target_data,
        validation_input_data,
        validation_target_data,
        holdout_input_data=None,
        holdout_target_data=None,
        test_input_data=None,
    ):
        PreprocessingPipelineMixIn.__init__(
            self,
            preprocessing_stage="intra_cv",
            pipeline=None,
            fitting_guide=None,
            preprocessing_params=None,
            features=features,
            target_column=target_column,
            train_input_data=train_input_data,
            train_target_data=train_target_data,
            validation_input_data=validation_input_data,
            validation_target_data=validation_target_data,
            holdout_input_data=holdout_input_data,
            holdout_target_data=holdout_target_data,
            test_input_data=test_input_data,
            fail_gracefully=False,
        )


class Sampler:
    def __init__(self, parameters, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data
        self.parameters = parameters

    def execute_pipeline(self):
        default_element = dict(
            method="", target_feature="target", target_value=-1.0, parameters=dict()
        )
        if len(self.parameters) > 0:
            self.report_status()

            for element in self.parameters:
                default_element.update(element)

                self.advance_pipeline(default_element)

    def advance_pipeline(self, element):
        # FLAG: imblearn functions return ndarray when give pandas series - FIGURE THAT SHIT OUT
        # FLAG: imblearn functions return ndarray when give pandas series - FIGURE THAT SHIT OUT
        # FLAG: imblearn functions return ndarray when give pandas series - FIGURE THAT SHIT OUT
        # FLAG: imblearn functions return ndarray when give pandas series - FIGURE THAT SHIT OUT

        if element["method"] == "smote":
            self.input_data, self.target_data = SMOTE(**element["parameters"]).fit_sample(
                self.input_data, self.target_data
            )
        elif element["method"] == "adasyn":
            self.input_data, self.target_data = ADASYN(**element["parameters"]).fit_sample(
                self.input_data, self.target_data
            )
        elif element["method"] == "RandomOverSampler":
            self.input_data, self.target_data = RandomOverSampler(
                **element["parameters"]
            ).fit_sample(self.input_data, self.target_data)

        # FLAG: imblearn functions return ndarray when give pandas series - FIGURE THAT SHIT OUT
        # FLAG: imblearn functions return ndarray when give pandas series - FIGURE THAT SHIT OUT
        # FLAG: imblearn functions return ndarray when give pandas series - FIGURE THAT SHIT OUT
        # FLAG: imblearn functions return ndarray when give pandas series - FIGURE THAT SHIT OUT

        elif element["method"] == "upsample":
            self.input_data, self.target_data = upsample(
                self.input_data,
                self.target_data,
                element["target_feature"],
                element["target_value"],
                **element["parameters"]
            )

        self.report_status(element["method"])

    def report_status(self, method=None):
        if method is None:
            print("Target Label Counts... {}".format(sorted(Counter(self.target_data).items())))
        else:
            print(
                "After Performing... {}... Target Label Counts... {}".format(
                    method, sorted(Counter(self.target_data).items())
                )
            )


# def _execute():
#     """EXPERIMENTAL"""
#     test_parameters = [
#         dict(
#             target_feature='target',
#             target_value=1.0,
#             method='smote'
#         ),
#         dict(
#             target_feature='target',
#             method='upsample',
#             target_value=1.0,
#             parameters=dict(
#                 n_times=3
#             )
#         ),
#         # dict(
#         #     target_feature='target',
#         #     target_value=1.0,
#         #     method='smote'
#         # ),
#         # dict(
#         #     target_feature='target',
#         #     target_value=1.0,
#         #     method='adasyn'
#         # )
#     ]
#
#     # train_data = pd.read_csv('./data/porto_seguro_train.csv')
#     # train_input = train_data.drop(['id', 'target'], axis=1)
#     # train_target = train_data['target']
#
#     test_sampler = Sampler(test_parameters, train_input, train_target)
#     # test_sampler.report_status()
#     # test_sampler.advance_pipeline()
#     test_sampler.execute_pipeline()
#
#     print('hold')
#
#
# if __name__ == '__main__':
#     _execute()
