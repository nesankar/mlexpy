import numpy as np
import logging
from joblib import dump, load
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Iterable, Callable, Union

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
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
)

from mlexpy.pipeline_utils import MLSetup, ExperimentSetup, cv_report
from mlexpy.utils import make_directory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExperimentBase:
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
        self.processor = None
        self.test_cv_split = 0.4
        self.rnd = np.random.RandomState(rnd_int)
        self.cv_split_count = cv_split_count
        self.metric_dict: Dict[str, Callable] = {}
        self.standard_metric = None
        self.process_tag = process_tag
        self.model_tag = model_tag

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
        if not self.model_dir.is_dir():
            make_directory(self.model_dir)

    def process_data(self, process_method_str: str = "process_data") -> ExperimentSetup:
        raise NotImplementedError("This needs to be implemented in the child class.")

    def train_model(
        self,
        model: Any,
        full_setup: ExperimentSetup,
        params: Optional[Dict[str, Any]] = None,
    ):
        if params:
            model = self.cv_search(
                full_setup.train_data,
                model,
                params,
            )
        else:
            logger.info("Performing standard model training.")
            model.fit(full_setup.train_data.obs, full_setup.train_data.labels)

        logger.info("Model trained")
        return model

    def predict(
        self, full_setup: ExperimentSetup, model: Any, proba: bool = False
    ) -> Any:
        if proba:
            return model.predict_proba(full_setup.test_data.obs)
        else:
            return model.predict(full_setup.test_data.obs)

    def evaluate_predictions(
        self,
        full_setup: ExperimentSetup,
        predictions: Iterable,
        class_probabilities: Optional[Iterable] = None,
        baseline_prediction: bool = False,
    ) -> Dict[str, float]:
        raise NotImplementedError("This needs to be implemented in the child class.")

    def cv_splits(self, n_splits: int = 5) -> StratifiedShuffleSplit:
        """Creates an object to be passed to cv_eval, allowing for
        # identical splits everytime cv_eval is used.
        """
        return StratifiedShuffleSplit(
            n_splits=n_splits, test_size=self.test_cv_split, random_state=self.rnd
        )

    def cv_search(
        self,
        data_setup: MLSetup,
        ml_model: Any,
        parameters: Dict[str, Any],
        cv_model: str = "random_search",
        random_iterations: int = 5,
    ) -> Any:
        """Run grid cross_validation search over the parameter space.
        If no GirdSearch model provided run random search
        """

        if not self.standard_metric:
            raise NotImplementedError(
                "No standard_metric has been set. This is likely because the ExperimentBase is being called, instead of being inherited. Try using the ClassifierExpirament or RegressionExpirament, or build a child class to inherit the ExpiramentBase."
            )

        if cv_model == "grid_search":
            cv_search = GridSearchCV(
                ml_model,
                parameters,
                scoring=self.standard_metric,
                cv=self.cv_splits(self.cv_split_count),
                n_jobs=1,
            )
        else:
            cv_search = RandomizedSearchCV(
                ml_model,
                parameters,
                n_iter=random_iterations,
                scoring=self.standard_metric,
                cv=self.cv_splits(self.cv_split_count),
                verbose=2,
                refit=True,
                n_jobs=1,
            )
        logger.info(f"Beginning CV search using {cv_model} ...")
        cv_search.fit(data_setup.obs, data_setup.labels)
        logger.info(cv_report(cv_search.cv_results_))
        return cv_search.best_estimator_

    def add_metric(self, metric: Callable, name: str) -> None:
        """Add the provided metric to the metric_dict"""
        self.metric_dict[name] = metric

    def remove_metric(self, name: str) -> None:
        """Add the provided metric to the metric_dict"""
        del self.metric_dict[name]

    def default_store_model(self, model: Any) -> None:
        """Given a calculated model, store it locally using joblib.
        Longer term/other considerations can be found here: https://scikit-learn.org/stable/model_persistence.html
        """
        if hasattr(model, "save_model"):
            # use the model's saving utilities, specifically beneficial wish xgboost. Can be beneficial here to use a json
            logger.info(f"Found a save_model method in {model}")
            model_path = self.model_dir / f"{self.model_tag}.json"
            model.save_model(model_path)
        else:
            logger.info(f"Saving the {model} model using joblib.")
            model_path = self.model_dir / f"{self.model_tag}.joblib"
            dump(model, model_path)
        logger.info(f"Dumped {self.model_tag} to: {model_path}")

    def default_load_model(self, model: Optional[Any] = None) -> Any:
        """Given a model name, load it from storage."""

        if hasattr(model, "load_model") and model:
            # use the model's loading utilities -- specifically beneficial with xgboost
            logger.info(f"Found a load_model method in {model}")
            model_path = self.model_dir / f"{self.model_tag}.json"
            logger.info(f"Loading {self.model_tag} from: {model_path}")
            loaded_model = model.load_model(model_path)
        else:
            model_path = self.model_dir / f"{self.model_tag}.joblib"
            logger.info(f"Loading {self.model_tag} from: {model_path}")
            loaded_model = load(model_path)
        logger.info(f"Retrieved {self.model_tag} from: {model_path}")
        return loaded_model


class ClassifierExperimentBase(ExperimentBase):
    def __init__(
        self,
        train_setup: MLSetup,
        test_setup: MLSetup,
        cv_split_count: int,
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
        self.baseline_value = None  # to be implemented in the child class
        self.standard_metric = balanced_accuracy_score
        self.metric_dict = {
            "f1": f1_score,
            "log_loss": log_loss,
            "balanced_accuracy": balanced_accuracy_score,
            "accuracy": accuracy_score,
            "confusion_matrix": confusion_matrix,
            "classification_report": classification_report,
        }

    def evaluate_predictions(
        self,
        full_setup: ExperimentSetup,
        predictions: Iterable,
        class_probabilities: Optional[Iterable] = None,
        baseline_prediction: bool = False,
    ) -> Dict[str, float]:
        """Evaluate all predictions, and return the results in a dict"""

        if baseline_prediction:
            if not self.baseline_value:
                raise ValueError(
                    "No baseline value was provided to the class and a baseline evaluation was called. Either set a baseline value or pass baseline_prediction=False to evaluate_predictions method."
                )
            evaluation_prediction = self.baseline_value
        else:
            evaluation_prediction = predictions

        result_dict: Dict[str, float] = {}
        # First test the predictions in the metric dictionary...
        for name, metric in self.metric_dict.items():
            if "f1" in name:
                result_dict[name + "_macro"] = metric(
                    full_setup.test_data.labels, evaluation_prediction, average="macro"
                )
                result_dict[name + "_micro"] = metric(
                    full_setup.test_data.labels, evaluation_prediction, average="micro"
                )
                result_dict[name + "_weighted"] = metric(
                    full_setup.test_data.labels,
                    evaluation_prediction,
                    average="weighted",
                )
            else:
                try:
                    result_dict[name] = metric(
                        full_setup.test_data.labels, evaluation_prediction
                    )
                except ValueError:
                    try:
                        result_dict[name] = metric(
                            full_setup.test_data.labels, class_probabilities
                        )
                    except ValueError:
                        print(f"Unknown issues with the {name} metric evaluation.")

        for name, score in result_dict.items():
            print(f"\nThe {name} score is: \n {score}.")

        return result_dict

    def evaluate_roc_metrics(
        self,
        full_setup: ExperimentSetup,
        model: Any,
    ) -> Dict[str, float]:
        """Perform any roc metric evaluation here. These require prediction probabilities or confidence, thus are separate
        from more standard prediction value based metrics."""

        probabilities = model.predict_proba(full_setup.test_data.obs)
        result_dict: Dict[str, float] = {}
        # Need to determine if using a multiclass or binary classification experiment
        if len(set(full_setup.test_data.labels)) <= 2:
            logger.info("Computing the binary AU ROC curve scores.")
            # Then this is binary classification
            result_dict["roc_auc_score"] = roc_auc_score(
                y_true=full_setup.test_data.labels,
                y_score=probabilities,
            )
            print(f"""\nThe ROC AUC score is: {result_dict["roc_auc_score"]}""")
        else:
            logger.info("Computing the multi-class AU ROC curve scores.")
            # We are doing multiclass classification and need to use more parameters to calculate the roc
            result_dict["roc_auc_score"] = roc_auc_score(
                y_true=full_setup.test_data.labels,
                y_score=probabilities,
                average="weighted",
                multi_class="ovo",
            )
            print(
                f"""\nThe multi-class weighted ROC AUC score is: {result_dict["roc_auc_score"]}"""
            )

        RocCurveDisplay.from_estimator(
            estimator=model,
            X=full_setup.test_data.obs,
            y=full_setup.test_data.labels,
        )
        return result_dict


class RegressionExperimentBase(ExperimentBase):
    def __init__(
        self,
        train_setup: MLSetup,
        test_setup: MLSetup,
        cv_split_count: int,
        rnd_int: int = 100,
        model_tag: str = "",
        model_storage_function: Optional[Callable] = None,
        model_loading_function: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            train_setup,
            test_setup,
            cv_split_count,
            rnd_int,
            model_tag,
            model_storage_function,
            model_loading_function,
        )
        self.baseline_value = None
        self.standard_metric = balanced_accuracy_score
        self.metric_dict = {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
        }

    def evaluate_predictions(
        self,
        full_setup: ExperimentSetup,
        predictions: Iterable,
        class_probabilities: Optional[Iterable] = None,
        baseline_prediction: bool = False,
    ) -> Dict[str, float]:
        """Evaluate all predictions, and return the results in a dict"""

        if baseline_prediction:
            if not self.baseline_value:
                raise ValueError(
                    "No baseline value was provided to the class and a baseline evaluation was called. Either set a baseline value or pass baseline_prediction=False to evaluate_predictions method."
                )
            evaluation_prediction = self.baseline_value
        else:
            evaluation_prediction = predictions

        result_dict: Dict[str, float] = {}
        # First test the predictions in the metric dictionary...
        for name, metric in self.metric_dict.items():
            result_dict[name] = metric(
                full_setup.test_data.labels, evaluation_prediction
            )
            print(f"\nThe {name} is: {result_dict[name]}")
        return result_dict
