import pytest
from mlexpy import experiment, processor, pipeline_utils
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Callable
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier
import sys
from numpy.testing import assert_array_equal
from fixtures import simple_dataframe, simple_binary_dataframe, rs_10, rs_20


@pytest.fixture
def basic_processor():
    class basic_processor(processor.ProcessPipelineBase):
        def __init__(
            self,
            process_tag: str = "_development",
            model_dir: Optional[Union[str, Path]] = None,
            model_storage_function: Optional[Callable] = None,
            model_loading_function: Optional[Callable] = None,
        ) -> None:
            super().__init__(
                process_tag, model_dir, model_storage_function, model_loading_function
            )

        def process_data(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:

            # First create a new feature
            df["new_feature"] = df["obs1"] * df["obs2"]

            # Now add a standard scaler...
            if training:
                self.fit_model_based_features(df)
                model_features = self.transform_model_based_features(df)
            else:
                # Here we can ONLY apply the transformation
                model_features = self.transform_model_based_features(df)

            return model_features

        def fit_model_based_features(self, df: pd.DataFrame) -> None:
            for column in df.columns:
                if not self.check_numeric_column(df[column]):
                    continue
                # Make the column retainment logic a bit complicated.
                if "length" in column:
                    drop_column = True
                else:
                    drop_column = False
                self.fit_scaler(
                    df[column], standard_scaling=True, drop_columns=drop_column
                )

    return basic_processor


@pytest.fixture
def regression_experiment():
    class regression_obj(experiment.RegressionExperimentBase):
        def __init__(
            self,
            train_setup: pipeline_utils.MLSetup,
            test_setup: pipeline_utils.MLSetup,
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

        def process_data(
            self, process_method_str: str = "process_data", from_file: bool = False
        ) -> pipeline_utils.ExperimentSetup:

            # Now get the the data processing method defined in process_method_str.
            process_method = getattr(self.pipeline, process_method_str)

            # First, determine if we are processing data via loading previously trained transformation models...
            if from_file:
                # ... if so, just perform the process_method function for training
                test_df = process_method(self.testing.obs, training=False)

                return pipeline_utils.ExperimentSetup(
                    pipeline_utils.MLSetup(
                        pd.DataFrame(),
                        pd.Series(),
                    ),
                    pipeline_utils.MLSetup(
                        test_df,
                        self.testing.labels,
                    ),
                )
            else:
                train_df = process_method(self.training.obs, training=True)
                test_df = process_method(self.testing.obs, training=False)

            print(
                f"The train data are of size {train_df.shape}, the test data are {test_df.shape}."
            )

            assert (
                len(set(train_df.index).intersection(set(test_df.index))) == 0
            ), "There are duplicated indices in the train and test set."

            return pipeline_utils.ExperimentSetup(
                pipeline_utils.MLSetup(
                    train_df,
                    self.training.labels,
                ),
                pipeline_utils.MLSetup(
                    test_df,
                    self.testing.labels,
                ),
            )

    return regression_obj


@pytest.fixture
def classification_experiment():
    class classification_obj(experiment.ClassifierExperimentBase):
        def __init__(
            self,
            train_setup: pipeline_utils.MLSetup,
            test_setup: pipeline_utils.MLSetup,
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

        def process_data(
            self, process_method_str: str = "process_data", from_file: bool = False
        ) -> pipeline_utils.ExperimentSetup:

            # Now get the the data processing method defined in process_method_str.
            process_method = getattr(self.pipeline, process_method_str)

            # First, determine if we are processing data via loading previously trained transformation models...
            if from_file:
                # ... if so, just perform the process_method function for training
                test_df = process_method(self.testing.obs, training=False)
                test_labels = self.pipeline.encode_labels(self.testing.labels)

                return pipeline_utils.ExperimentSetup(
                    pipeline_utils.MLSetup(
                        pd.DataFrame(),
                        pd.Series(),
                    ),
                    pipeline_utils.MLSetup(
                        test_df,
                        test_labels,
                    ),
                )
            else:
                train_df = process_method(self.training.obs, training=True)
                test_df = process_method(self.testing.obs, training=False)

                train_labels = self.pipeline.encode_labels(self.training.labels)
                test_labels = self.pipeline.encode_labels(self.testing.labels)

            print(
                f"The train data are of size {train_df.shape}, the test data are {test_df.shape}."
            )

            assert (
                len(set(train_df.index).intersection(set(test_df.index))) == 0
            ), "There are duplicated indices in the train and test set."

            return pipeline_utils.ExperimentSetup(
                pipeline_utils.MLSetup(
                    train_df,
                    train_labels,
                ),
                pipeline_utils.MLSetup(
                    test_df,
                    test_labels,
                ),
            )

    return classification_obj


def test_regression_model_match(
    simple_dataframe, rs_10, rs_20, regression_experiment, basic_processor
):

    # Now do the experiment...

    dataset = pipeline_utils.get_stratified_train_test_data(
        simple_dataframe[["obs1", "obs2"]],
        simple_dataframe["target"],
        test_frac=0.5,
        random_state=rs_20,
        stratify=False,
    )
    experiment_obj = regression_experiment(
        train_setup=dataset.train_data,
        test_setup=dataset.test_data,
        model_tag="regression_match_model",
        process_tag="regression_match_process",
    )

    experiment_obj.set_pipeline(basic_processor)

    # Now begin the experimentation, start with performing the data processing...
    processed_datasets = experiment_obj.process_data()

    # ... then train our model...
    trained_model = experiment_obj.train_model(
        RandomForestRegressor(random_state=rs_10),
        processed_datasets,
    )
    # Get the predictions and evaluate the performance.
    predictions = experiment_obj.predict(processed_datasets, trained_model)
    results = experiment_obj.evaluate_predictions(
        processed_datasets.test_data.labels,
        predictions=predictions,
    )

    # Here we store the model
    experiment_obj.store_model(trained_model)

    # Now, test that the results are the same if loading from a disk.
    loaded_datasets = experiment_obj.process_data_from_stored_models()
    loaded_model = experiment_obj.load_model()
    # Get the predictions and evaluate the performance.
    loaded_predictions = experiment_obj.predict(loaded_datasets, loaded_model)
    loaded_results = experiment_obj.evaluate_predictions(
        loaded_datasets.test_data.labels,
        predictions=loaded_predictions,
    )

    # Assert that the predictions are the same
    assert_array_equal(predictions, loaded_predictions)

    # Assert that the evaluation metrics are all the same
    assert all(
        [
            trained_value == loaded_results[trained_key]
            for trained_key, trained_value in results.items()
        ]
    )

    # Do the same thing with a newly named class
    new_experiment = regression_experiment(
        train_setup=dataset.train_data,
        test_setup=dataset.test_data,
        model_tag="regression_match_model",
        process_tag="regression_match_process",
    )
    new_experiment.set_pipeline(basic_processor)
    new_datasets = new_experiment.process_data_from_stored_models()
    new_model = new_experiment.load_model()

    # Get the predictions and evaluate the performance.
    new_predictions = new_experiment.predict(new_datasets, new_model)
    new_results = new_experiment.evaluate_predictions(
        new_datasets.test_data.labels,
        predictions=new_predictions,
    )

    # Assert that the predictions are the same
    assert_array_equal(predictions, new_predictions)

    # Assert that the evaluation metrics are all the same
    assert all(
        [
            trained_value == new_results[trained_key]
            for trained_key, trained_value in results.items()
        ]
    )


def test_classification_model_match(
    simple_binary_dataframe, rs_10, rs_20, classification_experiment, basic_processor
):

    # Now do the experiment...
    dataset = pipeline_utils.get_stratified_train_test_data(
        simple_binary_dataframe[["obs1", "obs2"]],
        simple_binary_dataframe["target"],
        test_frac=0.5,
        random_state=rs_20,
    )
    experiment_obj = classification_experiment(
        train_setup=dataset.train_data,
        test_setup=dataset.test_data,
        model_tag="classification_match_model",
        process_tag="classification_match_process",
    )

    experiment_obj.set_pipeline(basic_processor)

    # Now begin the experimentation, start with performing the data processing...
    processed_datasets = experiment_obj.process_data()

    # ... then train our model...
    trained_model = experiment_obj.train_model(
        XGBClassifier(random_state=rs_10),
        processed_datasets,
    )
    # Get the predictions and evaluate the performance.
    predictions = experiment_obj.predict(processed_datasets, trained_model)
    results = experiment_obj.evaluate_predictions(
        processed_datasets.test_data.labels,
        predictions=predictions,
    )

    # Here we store the model
    experiment_obj.store_model(trained_model)

    # Now, test that the results are the same if loading from a disk.
    loaded_datasets = experiment_obj.process_data_from_stored_models()
    loaded_model = experiment_obj.load_model(XGBClassifier())
    # Get the predictions and evaluate the performance.

    loaded_predictions = experiment_obj.predict(loaded_datasets, loaded_model)
    loaded_results = experiment_obj.evaluate_predictions(
        loaded_datasets.test_data.labels,
        predictions=loaded_predictions,
    )

    # Assert that the predictions are the same
    assert_array_equal(predictions, loaded_predictions)

    # Assert that the evaluation metrics are all the same
    assert all(
        [
            results[key] == loaded_results[key]
            for key in [
                "f1_micro",
                "f1_macro",
                "log_loss",
                "accuracy",
                "balanced_accuracy",
                "f1_weighted",
            ]
        ]
    )

    # Do the same thing with a newly named class
    new_experiment = classification_experiment(
        train_setup=dataset.train_data,
        test_setup=dataset.test_data,
        model_tag="classification_match_model",
        process_tag="classification_match_process",
    )
    new_experiment.set_pipeline(basic_processor)
    new_datasets = new_experiment.process_data_from_stored_models()
    new_model = new_experiment.load_model(XGBClassifier())

    # Get the predictions and evaluate the performance.
    new_predictions = new_experiment.predict(new_datasets, new_model)
    new_results = new_experiment.evaluate_predictions(
        new_datasets.test_data.labels,
        predictions=new_predictions,
    )

    # Assert that the predictions are the same
    assert_array_equal(predictions, new_predictions)

    # Assert that the evaluation metrics are all the same
    print(
        [
            results[key] == new_results[key]
            for key in [
                "f1_micro",
                "f1_macro",
                "log_loss",
                "accuracy",
                "balanced_accuracy",
                "f1_weighted",
            ]
        ]
    )
    assert all(
        [
            results[key] == new_results[key]
            for key in [
                "f1_micro",
                "f1_macro",
                "log_loss",
                "accuracy",
                "balanced_accuracy",
                "f1_weighted",
            ]
        ]
    )

    # Test that I can create probabs...
    probabilities = experiment_obj.predict(
        processed_datasets, trained_model, proba=True
    )
    # ... and that they can be evaluated...
    experiment_obj.evaluate_predictions(
        processed_datasets.test_data.labels,
        predictions=predictions,
        class_probabilities=probabilities,
    )

    # ... also evaluate the ROC based metrics WITHOUT ANY EXCEPTIONS.
    experiment_obj.evaluate_roc_metrics(
        processed_datasets, probabilities, trained_model
    )

    experiment_obj.plot_multiclass_roc(
        processed_datasets.test_data.labels, probabilities
    )
