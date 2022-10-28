import numpy as np
import logging
from typing import Dict, Optional, Any, Iterable

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
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
        self, full_setup: ExperimentSetup, predictions: Iterable
    ) -> None:
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

        if classify:
            metric = "balanced_accuracy"
        else:
            metric = "mean_absolute_error"

        if cv_model == "grid_search":
            cv_search = GridSearchCV(
                ml_model,
                parameters,
                scoring=metric,
                cv=self.cv_splits(self.cv_split_count),
                n_jobs=1,
            )
        else:
            cv_search = RandomizedSearchCV(
                ml_model,
                parameters,
                n_iter=random_iterations,
                scoring=metric,
                cv=self.cv_splits(self.cv_split_count),
                verbose=2,
                refit=True,
                n_jobs=1,
            )
        logger.info(f"Begining CV search using {cv_model} ...")
        cv_search.fit(data_setup.obs, data_setup.labels)
        logger.info(cv_report(cv_search.cv_results_))
        return cv_search.best_estimator_


class ClassifierExpirament(ExpiramentBase):
    def __init__(
        self, train_setup: MLSetup, test_setup: MLSetup, cv_split_count: int
    ) -> None:
        super().__init__(train_setup, test_setup, cv_split_count)
        self.baseline_value = None
        self.standard_metric = balanced_accuracy_score

    def process_data(self) -> ExperimentSetup:
        """Perform the data processing here"""
        raise NotImplementedError(
            "Needs to be implemented by the child, case specific expirament class."
        )

    def evaluate_predictions(
        self, full_setup: ExperimentSetup, predictions: Iterable
    ) -> None:
        """Evaluate all predictions"""

        # First test the predictions...
        conf_mat = confusion_matrix(full_setup.test_data.labels, predictions)
        bal_acc = balanced_accuracy_score(full_setup.test_data.labels, predictions)
        macro_f1 = f1_score(full_setup.test_data.labels, predictions, average="macro")
        micro_f1 = f1_score(full_setup.test_data.labels, predictions, average="micro")
        weighted_f1 = f1_score(
            full_setup.test_data.labels, predictions, average="weighted"
        )

        print(f"\nThe balanced_accuracy is: {bal_acc}")
        print(f"The F1 MICRO score is: {micro_f1}")
        print(f"The F1 MACRO score is: {macro_f1}")
        print(f"The F1 WEIGHTED score is: {weighted_f1}")
        print(f"Confusion matrix is:\n {conf_mat}")

        # ... then test the baseline if set.
        if self.baseline_value:
            baseline = [self.baseline_value] * len(predictions)

            baseline_conf_mat = confusion_matrix(full_setup.test_data.labels, baseline)
            baseline_bal_acc = balanced_accuracy_score(
                full_setup.test_data.labels, baseline
            )
            baseline_macro_f1 = f1_score(
                full_setup.test_data.labels, baseline, average="macro"
            )
            baseline_micro_f1 = f1_score(
                full_setup.test_data.labels, baseline, average="micro"
            )
            baseline_weighted_f1 = f1_score(
                full_setup.test_data.labels, baseline, average="weighted"
            )

            print(f"\n\nThe BASELINE balanced_accuracy is: {baseline_bal_acc}")
            print(f"The BASELINE F1 MICRO score is: {baseline_micro_f1}")
            print(f"The BASELINE F1 MACRO score is: {baseline_macro_f1}")
            print(f"The BASELINE F1 WEIGHTED score is: {baseline_weighted_f1}")
            print(f"BASELINE Confusion matrix is:\n {baseline_conf_mat}")
        else:
            logger.info(
                "No baseline_value was set, and so no baseling evaluation performed. To perform a baseline evaluation set self.baseline_value to your baseline."
            )
