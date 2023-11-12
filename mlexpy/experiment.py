import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import logging
from joblib import dump, load
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Callable, Union, Tuple, Iterable, List

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    mean_absolute_error,
    roc_auc_score,
    RocCurveDisplay,
    mean_squared_error,
    log_loss,
    auc,
    roc_curve,
)


from mlexpy.pipeline_utils import MLSetup, ExperimentSetup, CrossValidation, CVEval
from mlexpy.utils import make_directory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExperimentBase:
    """
    Base class to provide standard model experimentation tooling.

    Attributes
    ----------
    testing
        The test data MLSetup named tuple.
    training
        The train data MLSetup named tuple.
    test_cv_split
        The amount of data to use in each test set in cross validation. Only used if performing hyperparameter search.
    rnd
        An np.random.RandomState seed used to set the random seed for cv splitting, or random hyperparameter search.
    cv_split_count
        The number of splits to perform in any cv hyperparameter grid search.
    metric_dict
        A Dictionary of metrics to use to evaluate the model predictions.
    standard_cv_scorer
        The "standard cv metric" to use. This is what will be use in CV hyperparameter search as an loss function to minimize.
    process_tag
        A string to name the data processing methods. Used in naming the files that are dumped to disk.
    model_tag
        A string to define the model methods. Used in naming the files that are dumped to disk.
    pipeline
        The ProcessPipeline class to use to pre-process all data prior to modeling.
    standard_cv_scorer
        A callable attribute that sets how CV splits are defined, the default is an function returning None, but gets set as a StratifiedShuffleSplit downstream.

    Methods
    -------
    make_storage_dir()
       Used to create a directory at the class attribute model_dir.
    process_data(process_method_str: str = "process_data", from_file: bool = False)
        Method used to process the raw data. This needs to be defined in the experiment's child class that the user writes.
    process_data_from_stored_models()
        Method used to process raw data, however, used when loading all data processing from disk. Ex. if evaluating a new dataset with an old model.
    one_shot_train(self, ml_model: Any, data_setup: ExperimentSetup, parameters: Dict[str, List[Union[int, float, str]]]) -> Any:
        Method to call to train the model. If a params arg is not passed, then the default parameters are used.
    cv_train(self, ml_model: Any, data_setup: ExperimentSetup, parameters: Dict[str, List[Union[int, float, str]]], random_search: bool = True, random_iterations: int = 5, cv_split_function: Optional[Callable] = None) -> Any:
        Method to call to train the model using cross validation for hyperparameter search.
    predict(full_setup: ExperimentSetup, model: Any, proba: bool = False)
        The method to perform predictions for a model.
    cv_splits(n_splits: int = 5)
        A method to generate a cv-splitting method for cross validation.
    cv_search(data_setup: MLSetup, ml_model: Any, parameters: Dict[str, Any], cv_model: str = "random_search", random_iterations: int = 5)
        The method to perform cross-validated hyperparameter optimization. Currently, only grid or random search are options.
    add_metric(metric: Callable, name: str)
        Add a metric function to the metric_dict
    remove_metric(name: str)
        Remove a metric from the metric_dict
    default_store_model(model: Any, file_name: Optional[str] = None)
        Method to store a model. By default the model will be stored via joblib. Any model's native save method will be chosen here before joblib.
    default_load_model(model: Optional[Any] = None)
        Method to load a model. By default the model will be loaded via joblib. Any model's native load method will be chosen here before joblib.
    set_pipeline(self, pipeline: Type[ProcessPipelineBase], process_tag: Optional[str]=None)
        Method to set the current pipeline to use as the pre-processor.
    """

    def __init__(
        self,
        train_setup: MLSetup,
        test_setup: MLSetup,
        cv_split_count: int = 5,
        rnd_int: int = 100,
        model_dir: Optional[Union[str, Path]] = None,
        model_storage_function: Optional[Callable] = None,
        model_loading_function: Optional[Callable] = None,
        model_tag: str = "_development",
        process_tag: str = "_development",
    ) -> None:
        self.testing = test_setup
        self.training = train_setup
        self.test_cv_split = 0.4
        self.rnd = np.random.RandomState(rnd_int)
        self.cv_split_count = cv_split_count
        self.metric_dict: Dict[str, Callable] = {}
        self.process_tag = process_tag
        self.model_tag = model_tag
        self.pipeline: Any
        self.standard_cv_scorer: Callable = lambda: None

        # Setup model io
        if not model_storage_function:
            logger.info(
                "No model storage function provided. Using the default class method (joblib, or .store_model native method)."
            )
            self.store_model = self.default_store_model
        else:
            logger.info(f"Set the model storage function as: {model_storage_function}")
            self.store_model = model_storage_function
        if not model_loading_function:
            logger.info(
                "No model loading function provided. Using the default class method (joblib, or .load_model native method)."
            )
            self.load_model = self.default_load_model
        else:
            logger.info(f"Set the model loading function as: {model_loading_function}")
            self.store_model = model_loading_function

        if not model_dir:
            logger.info(
                f"No model location provided. Creating a .models/ at: {sys.path[-1]}"
            )
            self.model_dir = Path(sys.path[-1]) / ".models" / self.process_tag
        elif isinstance(model_dir, str):
            logger.info(
                f"setting the model path to {model_dir}. (Converting from string to pathlib.Path)"
            )
            self.model_dir = Path(model_dir) / self.process_tag
        else:
            logger.info(
                f"setting the model path to {model_dir}. (Converting from string to pathlib.Path)"
            )
            self.model_dir = model_dir / self.process_tag

    def make_storage_dir(self) -> None:
        """Create the directory of the mode_dir class attribute.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not self.model_dir.is_dir():
            make_directory(self.model_dir)

    def set_pipeline(self, pipeline: Any, process_tag: Optional[str] = None) -> None:
        """Set the pipeline attribute that is called in self.process_data to process the data for modeling.

        Parameters
        ----------
        pipeline : Type[ProcessPipelineBase]
            This is the user defined processor class (inheriting ProcessPipelineBase) used to process the raw data.
        process_tag : Optional[str]
            This the the string use to attach a name to the process models when they are stored to disk.

        Returns
        -------
        None
        """
        if not process_tag:
            process_tag = self.process_tag
        self.pipeline = pipeline(process_tag=process_tag, model_dir=self.model_dir)

    def process_data(
        self,
        process_method_str: str = "process_data",
        from_file: bool = False,
    ) -> ExperimentSetup:
        """Perform data processing here.

        Parameters
        ----------
        process_method_str : str
            The name of the method to use to process the data from the user defined ProcessPipeline class. By default is "process_data.
        from_file : bool
            A boolean flag to designate if the processing should be done via models from file ONLY.

        Returns
        -------
        ExperimentSetup An ExperimentSetup named tuple that contains the training and testing data to use to build a model over.
        """

        if not hasattr(self, "pipeline"):
            raise NameError(
                "The self.pipeline attribute has not be set. Run the .set_pipeline(<your-pipeline-class>) method to set the pipeline before processing."
            )

        # Now get the the data processing method defined in process_method_str.
        process_method = getattr(self.pipeline, process_method_str)

        # First, determine if we are processing data via loading previously trained transformation models...
        if from_file:
            # ... if so, just perform the process_method function for training
            if isinstance(self, ClassifierExperiment) and not is_numeric_dtype(
                self.training.labels
            ):
                test_labels = self.pipeline.encode_labels(labels=self.testing.labels)
            else:
                test_labels = self.testing.labels

            test_df = process_method(
                self.testing.obs, training=False, label_series=test_labels
            )

            return ExperimentSetup(
                MLSetup(
                    pd.DataFrame(),
                    pd.Series(),
                ),
                MLSetup(
                    test_df,
                    test_labels,
                ),
            )
        else:
            # Check if we might need to encode labels here
            if isinstance(self, ClassifierExperiment) and not is_numeric_dtype(
                self.training.labels
            ):
                # Then we need to encode the labels
                train_labels = self.pipeline.encode_labels(labels=self.training.labels)
                test_labels = self.pipeline.encode_labels(labels=self.testing.labels)
            else:
                train_labels = self.training.labels
                test_labels = self.testing.labels

            train_df = process_method(
                df=self.training.obs, training=True, label_series=train_labels
            )
            test_df = process_method(
                df=self.testing.obs, training=False, label_series=test_labels
            )

        print(
            f"The train data are of size {train_df.shape}, the test data are {test_df.shape}."
        )

        assert (
            len(set(train_df.index).intersection(set(test_df.index))) == 0
        ), "There are duplicated indices in the train and test set."

        return ExperimentSetup(
            MLSetup(
                train_df,
                train_labels,
            ),
            MLSetup(
                test_df,
                test_labels,
            ),
        )

    def process_data_from_stored_models(self) -> ExperimentSetup:
        """Perform data processing here, however only if using stored models.

        Parameters
        ----------
        None

        Returns
        -------
        An ExperimentSetup named tuple that contains the training and testing data to use to build a model over.
        """

        from_file_processed_data = self.process_data(from_file=True)
        return from_file_processed_data

    def one_shot_train(
        self,
        ml_model: Any,
        data_setup: ExperimentSetup,
        parameters: Dict[str, List[Union[int, float, str]]] = {},
    ) -> Any:
        """
        Do model training here.

        Parameters
        ----------
        ml_model : Any
            The machine learning model you would like to train. The only requirement is that the model have a .fit() method.
        data_setup : MLSetup
            The MLSetup object to train with. Contains the training and testing data.
        params : Optional[Dict[str, List[Union[int, float, str]]]]
            A dictionary storing the parameter name and values to use when training.

        Returns
        -------
        Any -- the model you have trained.
        """
        logger.info(
            f"Training over {data_setup.train_data.obs.shape[1]} features ({data_setup.train_data.obs.columns}) and {len(data_setup.train_data.obs)} examples."
        )
        logger.info(f"Performing standard model training for {ml_model}.")

        if any(
            (isinstance(value, Iterable) and not isinstance(value, str))
            for value in parameters.values()
        ):
            logger.warn(f"No lists allowed in the parameters. Check: {parameters}\n")
            raise (
                ValueError(
                    "One of the parameters passed to .one_shot_train() IS a list. If working over a variable parameter space use .cv_train(), otherwise, make sure the values in the params dict are all singe values, and not lists."
                )
            )

        ml_model = ml_model(**parameters)
        ml_model.fit(data_setup.train_data.obs, data_setup.train_data.labels)

        logger.info("Model trained")
        return ml_model

    def cv_train(
        self,
        ml_model: Any,
        data_setup: ExperimentSetup,
        parameters: Dict[str, List[Union[int, float, str]]],
        random_search: bool = True,
        random_iterations: int = 5,
        cv_split_function: Optional[Callable] = None,
    ) -> Any:
        """
        Perform cross-validated search over the hyperparameters for the best model parameters.

        Parameters
        ----------
        ml_model : Any
            The model you would like to train with hyperparameters tuned for.
        data_setup : MLSetup
            The MLSetup object to train with. Contains the training and testing data.
        parameters : Dict[str, Any]
            A dictionary defining the hyperparameter space to search over. Keys are parameter names, and values are spaces. Each key-value pair is a hyperparameter dimension.
        random_search: bool
            Argument to defined if to use random search or not. If not, a grid search is used.
        random_iterations : int
            The number of folds to use in the cross validation.
        cv_split_function : Optional[Callable]
            A function to  define how to perform the cross validation splitting. If not passed the default (StratifiedShuffleSplit) is used.

        Returns
        -------
        Any -- the trained model
        """

        logger.info(
            f"Training over {data_setup.train_data.obs.shape[1]} features ({data_setup.train_data.obs.columns}) and {len(data_setup.train_data.obs)} examples."
        )
        logger.info(f"Performing cross validated model training for {ml_model}.")

        if any(
            isinstance(value, Iterable) is False and isinstance(value, str) is True
            for value in parameters.values()
        ):
            logger.warn(
                f"No scalar values allowed in the parameters. Check: {parameters}\n"
            )
            raise (
                ValueError(
                    "One of the parameters passed to .one_shot_train() IS NOT a list. Can not search over the parameter space unless all values are lists of possible values. Note: a list of length 1 is valid"
                )
            )

        cv_searcher = CrossValidation(
            test_fraction=self.test_cv_split,
            score_function=self.standard_cv_scorer,
            n_splits=random_iterations,
            random_seed=self.rnd.get_state(legacy=False)["state"]["key"][
                -1
            ],  # needs to be an integer here
        )

        if cv_split_function:
            cv_searcher.set_split_function(cv_split_function)

        ml_model = cv_searcher.train_model(
            ml_model,
            data_setup.train_data,
            parameter_space=parameters,
            random_search=random_search,
            n_iterations=random_iterations,
        )

        logger.info("Model trained.")
        return ml_model

    def predict(
        self, data_setup: ExperimentSetup, ml_model: Any, proba: bool = False
    ) -> np.ndarray:
        """
        Do model prediction here.

        Parameters
        ----------
        data_setup : ExperimentSetup
            The ExperimentSetup named tuple that stores all dataset info for training and testing.
        ml_model : Any
            The model you would like to use for prediction. Must have a predict method.
        proba : bool
            A boolean flag to designate if probabilities should be returned. Only valid for classification problems, and if used the model must have a predict proba method.

        Returns
        -------
        nd.array
        """
        if proba:
            return ml_model.predict_proba(data_setup.test_data.obs)
        else:
            return ml_model.predict(data_setup.test_data.obs)

    def add_metric(self, metric: Callable, name: str) -> None:
        """
        Add a metric to the metric dict that is called in evaluation.

        Parameters
        ----------
        metric : Callable
            The number of splits (folds) created from the data in an iterations. Needs to accept (labels, predictions).

        name : str
            The name of the metric.

        Returns
        -------
        None
        """
        self.metric_dict[name] = metric

    def remove_metric(self, name: str) -> None:
        """Remove a metric to the metric dict that is called in evaluation.

        Parameters
        ----------
        name : str
            The name of the metric.

        Returns
        -------
        None
        """
        del self.metric_dict[name]

    def default_store_model(
        self, ml_model: Any, file_name: Optional[str] = None
    ) -> None:
        """Given a calculated model, store it locally using joblib.
        Longer term/other considerations can be found here: https://scikit-learn.org/stable/model_persistence.html

        Parameters
        ----------
        ml_model : Any
            The model you would like to store.

        file_name : Optional[str]
            The file name you would like. This string will precede the file type suffix.

        Returns
        -------
        None
        """
        self.make_storage_dir()

        if not file_name:
            file_name = self.model_tag

        if hasattr(ml_model, "save_model"):
            # use the model's saving utilities, specifically beneficial wish xgboost. Can be beneficial here to use a json
            logger.info(f"Found a save_model method in {ml_model}")
            model_path = self.model_dir / f"{file_name}.mdl"
            ml_model.save_model(model_path)
        else:
            logger.info(f"Saving the {ml_model} model using joblib.")
            model_path = self.model_dir / f"{file_name}.joblib"
            dump(ml_model, model_path)
        logger.info(f"Dumped {self.model_tag} to: {model_path}")

    def default_load_model(self, ml_model: Optional[Any] = None) -> Any:
        """Given a model name, load it from storage. The model is found by the model and process tag strings.

         Parameters
        ----------
        model : Optional[Any]
            The model you would like to store.

        Returns
        -------
        None
        """

        if hasattr(ml_model, "load_model") and ml_model:
            # use the model's loading utilities -- specifically beneficial with xgboost
            logger.info(f"Found a load_model method in {ml_model}")
            model_path = self.model_dir / f"{self.model_tag}.mdl"
            logger.info(f"Loading {self.model_tag} from: {model_path}")
            loaded_model = ml_model.load_model(model_path)
            if loaded_model is None:
                # Handle  case where the loaded model is instantiated in the provided model inplace
                return ml_model
        else:
            model_path = self.model_dir / f"{self.model_tag}.joblib"
            logger.info(f"Loading {self.model_tag} from: {model_path}")
            loaded_model = load(model_path)
        logger.info(f"Retrieved {self.model_tag} from: {model_path}")
        return loaded_model

    def evaluate_predictions_cross_validation(
        self,
        metric_function: Callable,
        predictions: np.ndarray,
        data: MLSetup,
        random_iterations: int,
    ) -> CVEval:
        """Given a trained model and an experiment setup, perform a cross validation of the evaluation.
        This is analogous to bootstrapping samples from the predictions, and then evaluating the metric
        on each bootstrapped sample.

        Parameters
        ----------
        metric_function : Callable
            The evaluation function you would like to use. It must accept the ground truth, then the predictions.
        predictions : np.ndarray
            The entire set of predictions made on the test set.
        data : MLSetup
            This is the MLSetup named tuple to evaluate against. Note it should be test data.
        random_iterations : int
            How many samples to generate.

        Return
        ------
        CVEval : A named tuple data structure of the result, "dot" indexed with mean, median, and std.
        """

        # First, setup the cross validation class
        cross_validator = CrossValidation(
            test_fraction=self.test_cv_split,
            score_function=self.standard_cv_scorer,
            n_splits=random_iterations,
            random_seed=self.rnd.get_state(legacy=False)["state"]["key"][
                -1
            ],  # needs to be an integer here
        )

        # Then do the cross validation evaluation
        result = cross_validator.validated_eval(
            data=data, predictions=predictions, metric=metric_function
        )
        print(
            f"\nThe {metric_function} cross validated scores are: \n mean: {result.mean}, median: {result.median}, standard deviation: {result.std}."
        )

        return result


class ClassifierExperiment(ExperimentBase):
    """
    Base class to provide standard model experimentation tooling for Classification problems.

    Attributes
    ----------
    testing
        The test data MLSetup named tuple.
    training
        The train data MLSetup named tuple.
    test_cv_split
        The amount of data to use in each test set in cross validation. Only used if performing hyperparameter search.
    rnd
        An np.random.RandomState seed used to set the random seed for cv splitting, or random hyperparameter search.
    cv_split_count
        The number of splits to perform in any cv hyperparameter grid search.
    metric_dict
        A Dictionary of metrics to use to evaluate the model predictions.
    standard_cv_scorer
        The "standard cv metric" to use. This is what will be use in CV hyperparameter search as an loss function to minimize.
    process_tag
        A string to name the data processing methods. Used in naming the files that are dumped to disk.
    model_tag
        A string to define the model methods. Used in naming the files that are dumped to disk.
    pipeline
        The ProcessPipeline class to use to pre-process all data prior to modeling.
    standard_cv_scorer
        A callable attribute that sets how CV splits are defined, the default is an function returning None, but gets set as a StratifiedShuffleSplit downstream.

    Methods
    -------
    make_storage_dir()
       Used to create a directory at the class attribute model_dir.
    process_data(process_method_str: str = "process_data", from_file: bool = False)
        Method used to process the raw data. This needs to be defined in the experiment's child class that the user writes.
    process_data_from_stored_models()
        Method used to process raw data, however, used when loading all data processing from disk. Ex. if evaluating a new dataset with an old model.
    one_shot_train(self, ml_model: Any, data_setup: ExperimentSetup, parameters: Dict[str, List[Union[int, float, str]]]) -> Any:
        Method to call to train the model. If a params arg is not passed, then the default parameters are used.
    cv_train(self, ml_model: Any, data_setup: ExperimentSetup, parameters: Dict[str, List[Union[int, float, str]]], random_search: bool = True, random_iterations: int = 5, cv_split_function: Optional[Callable] = None) -> Any:
        Method to call to train the model using cross validation for hyperparameter search.
    predict(full_setup: ExperimentSetup, model: Any, proba: bool = False)
        The method to perform predictions for a model.
    cv_splits(n_splits: int = 5)
        A method to generate a cv-splitting method for cross validation.
    cv_search(data_setup: MLSetup, ml_model: Any, parameters: Dict[str, Any], cv_model: str = "random_search", random_iterations: int = 5)
        The method to perform cross-validated hyperparameter optimization. Currently, only grid or random search are options.
    add_metric(metric: Callable, name: str)
        Add a metric function to the metric_dict
    remove_metric(name: str)
        Remove a metric from the metric_dict
    default_store_model(model: Any, file_name: Optional[str] = None)
        Method to store a model. By default the model will be stored via joblib. Any model's native save method will be chosen here before joblib.
    default_load_model(model: Optional[Any] = None)
        Method to load a model. By default the model will be loaded via joblib. Any model's native load method will be chosen here before joblib.
    set_pipeline(self, pipeline: Type[ProcessPipelineBase], process_tag: Optional[str]=None)
        Method to set the current pipeline to use as the pre-processor.
    evaluate_predictions(labels: Union[pd.Series, np.ndarray], predictions: Union[pd.Series, np.ndarray], class_probabilities: Optional[np.ndarray] = None, baseline_value: Optional[Union[float, int]] = None)
        Method to evaluate the predictions via classification metrics.
    evaluate_roc_metrics(full_setup: ExperimentSetup, class_probabilities: np.ndarray, model: Any)
        Method to evaluate specifically the AU_ROC curve metrics.
    plot_multiclass_roc(labels: Union[pd.Series, np.ndarray], class_probabilities: np.ndarray, fig_size: Tuple[int, int] = (8, 8))
        Method to plot ROC curves for multiclass (or binary) classification problems.
    """

    def __init__(
        self,
        train_setup: MLSetup,
        test_setup: MLSetup,
        cv_split_count: int = 5,
        rnd_int: int = 100,
        model_dir: Optional[Union[str, Path]] = None,
        model_storage_function: Optional[Callable] = None,
        model_loading_function: Optional[Callable] = None,
        model_tag: str = "_development",
        process_tag: str = "_development",
    ) -> None:
        super().__init__(
            train_setup,
            test_setup,
            cv_split_count,
            rnd_int,
            model_dir,
            model_storage_function,
            model_loading_function,
            model_tag,
            process_tag,
        )
        self.metric_dict = {
            "f1": f1_score,
            "log_loss": log_loss,
            "balanced_accuracy": balanced_accuracy_score,
            "accuracy": accuracy_score,
            "confusion_matrix": confusion_matrix,
            "classification_report": classification_report,
        }
        self.standard_cv_scorer = lambda labels, preds: -f1_score(
            labels, preds, average="macro"
        )

    def evaluate_predictions(
        self,
        labels: Union[pd.Series, np.ndarray],
        predictions: Union[pd.Series, np.ndarray],
        class_probabilities: Optional[np.ndarray] = None,
        baseline_value: Optional[Union[float, int]] = None,
    ) -> Dict[str, float]:
        """Evaluate all predictions, and return the results in a dict.

        Parameters
        ----------
        labels : Union[pd.Series, np.ndarray]
            The true class labels for the data.
        predictions : Union[pd.Series, np.ndarray]
            The class labels predicted from the model.
        class_probabilities : Optional[np.ndarray]
            The probability prediction of each class.
        baseline_value : Optional[Union[float, int]]
            If provided, will be used as a single value for every class. Ex. the most common class.

        Returns
        -------
        Dict[str, float]
        """

        if baseline_value:
            evaluation_prediction = np.repeat(baseline_value, len(labels))
        else:
            evaluation_prediction = predictions

        result_dict: Dict[str, float] = {}
        # First test the predictions in the metric dictionary...
        for name, metric in self.metric_dict.items():
            if "f1" in name:
                result_dict[name + "_macro"] = metric(
                    labels, evaluation_prediction, average="macro"
                )
                result_dict[name + "_micro"] = metric(
                    labels, evaluation_prediction, average="micro"
                )
                result_dict[name + "_weighted"] = metric(
                    labels,
                    evaluation_prediction,
                    average="weighted",
                )
            else:
                try:
                    result_dict[name] = metric(labels, evaluation_prediction)
                except ValueError:
                    # See if we would succeed with using the class probabilities
                    try:
                        result_dict[name] = metric(labels, class_probabilities)
                    except ValueError:
                        print(f"Unknown issues with the {name} metric evaluation.")

        for name, score in result_dict.items():
            print(f"\nThe {name} score is: \n {score}.")

        return result_dict

    def evaluate_roc_metrics(
        self,
        full_setup: ExperimentSetup,
        class_probabilities: np.ndarray,
        ml_model: Any,
    ) -> Dict[str, float]:
        """Perform any roc metric evaluation here. These require prediction probabilities or confidence, thus are separate
        from more standard prediction value based metrics.

        Parameters
        ----------
        full_setup : ExperimentSetup
            All data for the experiment. Because the AUC will be calculated from model.
        class_probabilities : Optional[np.ndarray]
            The probability prediction of each class.
        ml_model : Any
            The trained model from which the predictions will be evaluated.

        Returns
        -------
        Dict[str, float]
        """

        # First, check that there are more than 1 predictions
        if len(class_probabilities) <= 1:
            raise ValueError(
                f"The class_probabilities passed to evaluate_roc_metrics is only 1 record class_probabilities.shape = {class_probabilities.shape}"
            )

        result_dict: Dict[str, float] = {}
        # Need to determine if using a multiclass or binary classification experiment
        if len(class_probabilities[0]) <= 2:
            logger.info("Computing the binary AU-ROC curve scores.")
            # Then this is binary classification. Note from sklearn docs: The probability estimates correspond
            # to the **probability of the class with the greater label**
            result_dict["roc_auc_score"] = roc_auc_score(
                y_true=full_setup.test_data.labels,
                y_score=class_probabilities[:, 1],
            )
            print(f"""\nThe ROC AUC score is: {result_dict["roc_auc_score"]}""")

            dsp = RocCurveDisplay.from_estimator(
                estimator=ml_model,
                X=full_setup.test_data.obs,
                y=full_setup.test_data.labels,
            )
            dsp.plot()
            plt.show()

        else:
            logger.info("Computing the multi-class AU-ROC curve scores.")
            # We are doing multiclass classification and need to use more parameters to calculate the roc
            result_dict["roc_auc_score"] = roc_auc_score(
                y_true=full_setup.test_data.labels,
                y_score=class_probabilities,
                average="weighted",
                multi_class="ovr",
            )
            print(
                f"""\nThe multi-class weighted ROC AUC score is: {result_dict["roc_auc_score"]}"""
            )

            self.plot_multiclass_roc(
                labels=full_setup.test_data.labels,
                class_probabilities=class_probabilities,
            )

        return result_dict

    def plot_multiclass_roc(
        self,
        labels: Union[pd.Series, np.ndarray],
        class_probabilities: np.ndarray,
        fig_size: Tuple[int, int] = (8, 8),
    ) -> None:
        """Following from here: https://stackoverflow.com/questions/45332410/roc-for-multiclass-classification

        Parameters
        ----------
        labels : Union[pd.Series, np.ndarray]
            The true class labels for the data.
        class_probabilities : Optional[np.ndarray]
            The probability prediction of each class.
        fig_size : Tuple[int, int]
            The figure size parameter for a matplotlib plot.

        Returns
        -------
        Dict[str, float]
        """

        _, class_count = class_probabilities.shape

        fpr, tpr, roc_auc = {}, {}, {}

        # First, calculate all of the explicit class roc curves...
        y_test_dummies = pd.get_dummies(labels, drop_first=False).values
        for i in range(class_count):
            fpr[i], tpr[i], _ = roc_curve(
                y_test_dummies[:, i], class_probabilities[:, i]
            )
            roc_auc[i] = auc(fpr[i], tpr[i])

        #  Construct all plots
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver operating characteristic evaluation")
        for i in range(class_count):
            ax.plot(
                fpr[i],
                tpr[i],
                label=f"ROC curve (area = {round(roc_auc[i], 2)}) for label {i}",
                alpha=0.6,
            )

        ax.legend(loc="best")
        ax.grid(alpha=0.4)
        sns.despine()
        plt.show()


class RegressionExperiment(ExperimentBase):
    """
    Base class to provide standard model experimentation tooling for Classification problems.

    Attributes
    ----------
    testing
        The test data MLSetup named tuple.
    training
        The train data MLSetup named tuple.
    test_cv_split
        The amount of data to use in each test set in cross validation. Only used if performing hyperparameter search.
    rnd
        An np.random.RandomState seed used to set the random seed for cv splitting, or random hyperparameter search.
    cv_split_count
        The number of splits to perform in any cv hyperparameter grid search.
    metric_dict
        A Dictionary of metrics to use to evaluate the model predictions.
    standard_cv_scorer
        The "standard cv metric" to use. This is what will be use in CV hyperparameter search as an loss function to minimize.
    process_tag
        A string to name the data processing methods. Used in naming the files that are dumped to disk.
    model_tag
        A string to define the model methods. Used in naming the files that are dumped to disk.
    pipeline
        The ProcessPipeline class to use to pre-process all data prior to modeling.
    standard_cv_scorer
        A callable attribute that sets how CV splits are defined, the default is an function returning None, but gets set as a StratifiedShuffleSplit downstream.

    Methods
    -------
    make_storage_dir()
       Used to create a directory at the class attribute model_dir.
    process_data(process_method_str: str = "process_data", from_file: bool = False)
        Method used to process the raw data. This needs to be defined in the experiment's child class that the user writes.
    process_data_from_stored_models()
        Method used to process raw data, however, used when loading all data processing from disk. Ex. if evaluating a new dataset with an old model.
    one_shot_train(self, ml_model: Any, data_setup: ExperimentSetup, parameters: Dict[str, List[Union[int, float, str]]]) -> Any:
        Method to call to train the model. If a params arg is not passed, then the default parameters are used.
    cv_train(self, ml_model: Any, data_setup: ExperimentSetup, parameters: Dict[str, List[Union[int, float, str]]], random_search: bool = True, random_iterations: int = 5, cv_split_function: Optional[Callable] = None) -> Any:
        Method to call to train the model using cross validation for hyperparameter search.
    predict(full_setup: ExperimentSetup, model: Any, proba: bool = False)
        The method to perform predictions for a model.
    cv_splits(n_splits: int = 5)
        A method to generate a cv-splitting method for cross validation.
    cv_search(data_setup: MLSetup, ml_model: Any, parameters: Dict[str, Any], cv_model: str = "random_search", random_iterations: int = 5)
        The method to perform cross-validated hyperparameter optimization. Currently, only grid or random search are options.
    add_metric(metric: Callable, name: str)
        Add a metric function to the metric_dict
    remove_metric(name: str)
        Remove a metric from the metric_dict
    default_store_model(model: Any, file_name: Optional[str] = None)
        Method to store a model. By default the model will be stored via joblib. Any model's native save method will be chosen here before joblib.
    default_load_model(model: Optional[Any] = None)
        Method to load a model. By default the model will be loaded via joblib. Any model's native load method will be chosen here before joblib.
    set_pipeline(self, pipeline: Type[ProcessPipelineBase], process_tag: Optional[str]=None)
        Method to set the current pipeline to use as the pre-processor.
    evaluate_predictions(labels: Union[pd.Series, np.ndarray], predictions: Union[pd.Series, np.ndarray], class_probabilities: Optional[np.ndarray] = None, baseline_value: Optional[Union[float, int]] = None)
        Method to evaluate the predictions via regression metrics.
    """

    def __init__(
        self,
        train_setup: MLSetup,
        test_setup: MLSetup,
        cv_split_count: int = 5,
        rnd_int: int = 100,
        model_dir: Optional[Union[str, Path]] = None,
        model_storage_function: Optional[Callable] = None,
        model_loading_function: Optional[Callable] = None,
        model_tag: str = "_development",
        process_tag: str = "_development",
    ) -> None:
        super().__init__(
            train_setup,
            test_setup,
            cv_split_count,
            rnd_int,
            model_dir,
            model_storage_function,
            model_loading_function,
            model_tag,
            process_tag,
        )
        self.metric_dict = {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
        }
        self.standard_cv_scorer = lambda labels, preds: np.sqrt(
            mean_squared_error(labels, preds)
        )

    def evaluate_predictions(
        self,
        labels: Union[pd.Series, np.ndarray],
        predictions: Union[pd.Series, np.ndarray],
        baseline_value: Optional[Union[float, int]] = None,
    ) -> Dict[str, float]:
        """Evaluate all predictions, and return the results in a dict.

        Parameters
        ----------
        labels : Union[pd.Series, np.ndarray]
            The true class labels for the data.
        predictions : Union[pd.Series, np.ndarray]
            The class labels predicted from the model.
        baseline_value : Optional[Union[float, int]]
            If provided, will be used as a single value for every class. Ex. the most common class.

        Returns
        -------
        Dict[str, float]
        """

        if baseline_value:
            evaluation_prediction = np.repeat(baseline_value, len(labels))
        else:
            evaluation_prediction = predictions

        result_dict: Dict[str, float] = {}
        # First test the predictions in the metric dictionary...
        for name, metric in self.metric_dict.items():
            result_dict[name] = metric(labels, evaluation_prediction)
            print(f"\nThe {name} is: {result_dict[name]}")
        return result_dict
