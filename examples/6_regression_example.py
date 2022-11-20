import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Optional, Callable, Union
import argparse
from sklearn.datasets import load_diabetes

sys.path.append(str(Path.cwd()))
print(sys.path)
from mlexpy import pipeline_utils, experiment, processor
from examples.model_defs import MODEL_DICT


def parse_args(args: List[str]) -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--model_seed",
        help="What is the model random seed to use?",
        default=10,
        type=int,
    )

    parser.add_argument(
        "-p",
        "--process_seed",
        help="What is the process random seed to use?",
        default=20,
        type=int,
    )

    parser.add_argument(
        "-f",
        "--test_frac",
        help="What is the desired test ratio to use?",
        default=0.35,
        type=float,
    )

    parser.add_argument(
        "-m",
        "--model_type",
        help="What is the model  to use?",
        choices=["random_forest", "sgd"],
        type=str,
    )

    return parser.parse_args(args)


# Set up a custom metric, the averge prediction % error
def ave_abs_pct_error(labels: pd.Series, predictions: pd.Series) -> float:
    """For a given series of data, calcualte the average % error of every prediction."""

    def pairwise_pct_error(true: float, pred: float) -> float:
        if true == 0:
            print("The true value is 0. % error cannot be calcuated.")
            return np.nan

        return (abs(true - pred) / abs(true)) * 100

    if len(labels) != len(predictions):
        raise ValueError("There are not the same number of predictions and labels.")

    return sum(
        [
            pairwise_pct_error(labels.iloc[i], prediction)
            for i, prediction in enumerate(predictions)
        ]
    ) / len(labels)


# Define the processor class
class DiabetesProcessor(processor.ProcessPipelineBase):
    def __init__(
        self,
        process_tag: str = "_development",
        model_dir: Optional[Union[str, Path]] = None,
        model_storage_function: Optional[Callable] = None,
        model_loading_function: Optional[Callable] = None,
        store_models: bool = True,
    ) -> None:
        super().__init__(
            process_tag,
            model_dir,
            model_storage_function,
            model_loading_function,
            store_models,
        )

    def process_data(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:

        """All data processing that is to be performed for the iris classification task."""

        # Do a copy of the passed df
        df = df.copy()

        # First, compute the petal / sepal areas (but make the columns simpler)
        df.columns = [col.replace(" ", "_") for col in df.columns]

        return df


class DiabetesExperiment(experiment.RegressionExperimentBase):
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

        processor = DiabetesProcessor(
            process_tag=self.process_tag, model_dir=self.model_dir
        )

        # Now get the the data processing method defined in process_method_str.
        process_method = getattr(processor, process_method_str)

        # First, determine if we are processing data via loading previously trained transformation models...
        if from_file:
            # ... if so, just perform the process_method function for training
            test_df = process_method(self.testing.obs, training=False)

            # TODO: Also add loading a label encoder here...

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


# Now do the experimentation

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # First, set the random seed(s) for the experiment

    model_rs = np.random.RandomState(args.model_seed)
    process_rs = np.random.RandomState(args.process_seed)

    data = load_diabetes(as_frame=True)
    labels = data["target"]
    features = data["data"]

    experiment_setup = pipeline_utils.get_stratified_train_test_data(
        train_data=features,
        label_data=labels,
        test_frac=args.test_frac,
        random_state=process_rs,
    )
    # Define the experiment
    experiment_obj = DiabetesExperiment(
        train_setup=experiment_setup.train_data,
        test_setup=experiment_setup.test_data,
        cv_split_count=20,
        model_tag="example_development_model",
        process_tag="example_development_process",
    )

    # Now begin the experimentation, start with performing the data processing...
    processed_datasets = experiment_obj.process_data()

    # ... then train our model. Define our model based on what we have written into our model dictionary
    method_definition_function = MODEL_DICT["randomforest"]

    # First, call the method_definition_function with the classifier/regressor boolean defined...
    ml_model_data = method_definition_function(classifier=False)

    # Now we instantiate the model from the ml_model_data.model attribute...
    ml_model = ml_model_data.model(random_state=model_rs)

    # ... then train our model...
    trained_model = experiment_obj.train_model(
        ml_model,
        processed_datasets,
        # model_algorithm.hyperparams,  # If this is passed, then cross validation search is performed, but slow.
    )

    # Get the predictions and evaluate the performance.
    predictions = experiment_obj.predict(processed_datasets, trained_model)

    # Add our custom metric
    experiment_obj.add_metric(ave_abs_pct_error, "avg_abs_pct_error")

    results = experiment_obj.evaluate_predictions(
        processed_datasets.test_data.labels,
        predictions=predictions,
    )
