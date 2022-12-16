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

        def process_data(
            self,
            df: pd.DataFrame,
            training: bool = True,
            label_series: Optional[pd.Series] = None,
        ) -> pd.DataFrame:

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


def test_regression_model_match(simple_dataframe, rs_10, rs_20, basic_processor):

    # Now do the experiment...

    dataset = pipeline_utils.get_stratified_train_test_data(
        simple_dataframe[["obs1", "obs2"]],
        simple_dataframe["target"],
        test_frac=0.5,
        random_state=rs_20,
        stratify=False,
    )
    experiment_obj = experiment.RegressionExperiment(
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
    new_experiment = experiment.RegressionExperiment(
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
    simple_binary_dataframe, rs_10, rs_20, basic_processor
):

    # Now do the experiment...
    dataset = pipeline_utils.get_stratified_train_test_data(
        simple_binary_dataframe[["obs1", "obs2"]],
        simple_binary_dataframe["target"],
        test_frac=0.5,
        random_state=rs_20,
    )
    experiment_obj = experiment.ClassifierExperiment(
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
    new_experiment = experiment.ClassifierExperiment(
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
