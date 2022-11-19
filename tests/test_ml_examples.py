import pytest
from mlexpy import experiment, processor, pipeline_utils
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Callable
from sklearn.ensemble import RandomForestRegressor
import sys
from numpy.testing import assert_array_equal
from fixtures import simple_dataframe, to_scale_dataframe, rs_10, rs_20


def test_regression_model_match(simple_dataframe, rs_10, rs_20):

    # First, crete a processor object
    class my_processor(processor.ProcessPipelineBase):
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
                self.dump_feature_based_models()
            else:
                # Here we can ONLY apply the transformation
                model_features = self.transform_model_based_features(df)

            return model_features

        def fit_model_based_features(self, df: pd.DataFrame) -> None:
            for column in df.columns:
                if not self.check_numeric_column(df[column]):
                    print(df[column].dtype)
                    continue
                self.fit_scaler(df[column], standard_scaling=True)

    class my_experiment(experiment.RegressionExperimentBase):
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

            data_processor = my_processor(
                process_tag=self.process_tag, model_dir=self.model_dir
            )

            # Now get the the data processing method defined in process_method_str.
            process_method = getattr(data_processor, process_method_str)

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

    # Now do the experiment...

    dataset = pipeline_utils.get_stratified_train_test_data(
        simple_dataframe[["obs1", "obs2"]],
        simple_dataframe["target"],
        test_frac=0.5,
        random_state=rs_20,
    )
    experiment_obj = my_experiment(
        train_setup=dataset.train_data,
        test_setup=dataset.test_data,
        model_tag="test_regression_model_match_process",
        process_tag="test_regression_model_match_model",
        model_dir=Path(
            __file__
        ).parent,  # This means I will place the models alongside this script
    )

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
        processed_datasets,
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
        loaded_datasets,
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
    new_experiment = my_experiment(
        train_setup=dataset.train_data,
        test_setup=dataset.test_data,
        model_tag="test_regression_model_match_process",
        process_tag="test_regression_model_match_model",
        model_dir=Path(
            __file__
        ).parent,  # This means I will place the models alongside this script
    )
    new_datasets = new_experiment.process_data_from_stored_models()
    new_model = new_experiment.load_model()

    # Get the predictions and evaluate the performance.
    new_predictions = new_experiment.predict(new_datasets, new_model)
    new_results = new_experiment.evaluate_predictions(
        new_datasets,
        predictions=new_predictions,
    )

    # Assert that the predictions are the same
    assert_array_equal(predictions, loaded_predictions)

    # Assert that the evaluation metrics are all the same
    assert all(
        [
            trained_value == new_results[trained_key]
            for trained_key, trained_value in results.items()
        ]
    )
