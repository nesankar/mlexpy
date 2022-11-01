import numpy as np
import logging
from typing import Dict, Optional, Any, Iterable, Callable

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
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
)

from mlexpy.pipeline_utils import MLSetup, ExperimentSetup, cv_report


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExpiramentBase:
    def __init__(
        self,
        train_setup: MLSetup,
        test_setup: MLSetup,
        cv_split_count: int = 5,
        rnd_int=100,
    ) -> None:
        self.testing = test_setup
        self.training = train_setup
        self.processor = None
        self.test_cv_split = 0.4
        self.rnd = np.random.RandomState(rnd_int)
        self.cv_split_count = cv_split_count
        self.metric_dict: Dict[str, Callable] = {}
        self.standard_metric = None

    def set_pipeline(self, feat_coor_thresh: float = 0.90, top_cols=0.5) -> None:
        """Reset the params of the pipeline"""
        raise NotImplementedError("This needs to be implemented in the child class.")

    def process_data(self) -> ExperimentSetup:
        raise NotImplementedError("This needs to be implemented in the child class.")

    def train_model(
        self,
        model: Any,
        full_setup: ExperimentSetup,
        classify: bool,
        params: Optional[Dict[str, Any]] = None,
    ):
        if params:
            model = self.cv_search(full_setup.train_data, model, params, classify)
        else:
            logger.info("Performing standard model training.")
            model.fit(full_setup.train_data.obs, full_setup.train_data.labels)

        logger.info("Model trained")
        return model

    def predict(self, full_setup: ExperimentSetup, model: Any) -> Any:
        predictions = model.predict(full_setup.test_data.obs)
        return predictions

    def evaluate_predictions(
        self,
        full_setup: ExperimentSetup,
        predictions: Iterable,
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
        classify: bool = True,
        cv_model: str = "random_search",
        random_iterations: int = 5,
    ) -> Any:
        """Run grid cross_validation search over the parameter space.
        If no GirdSearch model provided run random search
        """

        if not self.standard_metric:
            raise NotImplementedError(
                "No standard_metric has been set. This is likely becuase the ExpiramentBase is being called, instead of being inherited. Try using the ClassifierExpirament or RegressionExpirament, or build a child class to inherit the ExpiramentBase."
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
        logger.info(f"Begining CV search using {cv_model} ...")
        cv_search.fit(data_setup.obs, data_setup.labels)
        logger.info(cv_report(cv_search.cv_results_))
        return cv_search.best_estimator_

    def add_metric(self, metric: Callable, name: str) -> None:
        """Add the provided metric to the metric_dict"""
        self.metric_dict[name] = metric

    def remove_metric(self, name: str) -> None:
        """Add the provided metric to the metric_dict"""
        del self.metric_dict[name]


class ClassifierExpirament(ExpiramentBase):
    def __init__(
        self, train_setup: MLSetup, test_setup: MLSetup, cv_split_count: int
    ) -> None:
        super().__init__(train_setup, test_setup, cv_split_count)
        self.baseline_value = None
        self.standard_metric = balanced_accuracy_score
        self.metric_dict = {
            "f1_macro": f1_score,
            "balanced_accruacry": balanced_accuracy_score,
            "accuracy": accuracy_score,
            "confusion_matrix": confusion_matrix,
            "classification_report": classification_report,
        }

    def process_data(self) -> ExperimentSetup:
        """Perform the data processing here"""
        raise NotImplementedError(
            "Needs to be implemented by the child, case specific expirament class."
        )

    def evaluate_predictions(
        self,
        full_setup: ExperimentSetup,
        predictions: Iterable,
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
                result_dict[name] = metric(
                    full_setup.test_data.labels, evaluation_prediction
                )

        for name, score in result_dict.items():
            print(f"\n\nThe {name} score is: \n {score}.")

        return result_dict

    def evaluate_roc_metrics(
        self,
        full_setup: ExperimentSetup,
        model: Any,
    ) -> Dict[str, float]:
        """Perform any roc metric evaluation here. These require prediction probilites or confidence, thus are seperate
        from more standand prediction value based metrics."""

        probabilites = model.predict_proba(full_setup.test_data.obs)
        result_dict: Dict[str, float] = {}
        # Need to determine if using a multiclass or binary classication expirament
        if len(set(full_setup.test_data.labels)) <= 2:
            logger.info("Computing the binary AU ROC curve scores.")
            # Then this is binary classiciation
            result_dict["roc_auc_score"] = roc_auc_score(
                y_true=full_setup.test_data.labels,
                y_score=probabilites,
            )
            print(f"""\nThe ROC AUC score is: {result_dict["roc_auc_score"]}""")
        else:
            logger.info("Computing the multi-class AU ROC curve scores.")
            # We are doing multiclass classicition and need to use more parameteres to calcualte the roc
            result_dict["roc_auc_score"] = roc_auc_score(
                y_true=full_setup.test_data.labels,
                y_score=probabilites,
                average="weighted",
                multi_class="ovo",
            )
            print(
                f"""\nThe multi-class weighted ROC AUC score is: {result_dict["roc_auc_score"]}"""
            )

        RocCurveDisplay.from_estimator(
            estimator=model,
            X=full_setup.test_data.obs,
            y=full_setup.test_data.lables,
        )
        return result_dict


class RegressionExpirament(ExpiramentBase):
    def __init__(
        self, train_setup: MLSetup, test_setup: MLSetup, cv_split_count: int
    ) -> None:
        super().__init__(train_setup, test_setup, cv_split_count)
        self.baseline_value = None
        self.standard_metric = balanced_accuracy_score
        self.metric_dict = {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
        }

    def process_data(self) -> ExperimentSetup:
        """Perform the data processing here"""
        raise NotImplementedError(
            "Needs to be implemented by the child, case specific expirament class."
        )

    def evaluate_predictions(
        self,
        full_setup: ExperimentSetup,
        predictions: Iterable,
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
