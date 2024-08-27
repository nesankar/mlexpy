import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Optional, Callable, Union
import argparse
from sklearn.datasets import load_diabetes
from mlexpy import pipeline_utils, experiment, processor
from model_defs import MODEL_DICT


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

    def process_data(
        self,
        df: pd.DataFrame,
        training: bool = True,
        label_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """All data processing that is to be performed for the diabetes prediction task."""

        # Do a copy of the passed df
        df = df.copy()

        df.columns = [col.replace(" ", "_") for col in df.columns]

        # Now perform the training / testing dependent feature processing. This is why a `training` boolean is passed.
        if training:
            # Now FIT all of the model based features...
            self.fit_model_based_features(df)
            # ... and get the results of a transformation of all model based features.
            model_features = self.transform_model_based_features(df)
        else:
            # Here we can ONLY apply the transformation
            model_features = self.transform_model_based_features(df)

        return model_features

    def fit_model_based_features(self, df: pd.DataFrame) -> None:

        standard_cols = ["age", "sex", "bmi", "bp"]
        non_standard_cols = [col for col in df.columns if col not in standard_cols]
        for col in standard_cols:
            # Just scale these columns
            self.fit_scaler(df[col], standard_scaling=True, drop_columns=True)

        self.fit_pca(df[non_standard_cols], n_components=3, drop_columns=True)


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
        stratify=False,
    )
    # Define the experiment
    experiment_obj = experiment.RegressionExperiment(
        train_setup=experiment_setup.train_data,
        test_setup=experiment_setup.test_data,
        cv_split_count=20,
        model_tag="example_development_model",
        process_tag="example_development_process",
        model_dir=Path(__file__).parent,
    )

    experiment_obj.set_pipeline(DiabetesProcessor)

    # Now begin the experimentation, start with performing the data processing...
    processed_datasets = experiment_obj.process_data()

    # ... then train our model. Define our model based on what we have written into our model dictionary
    method_definition_function = MODEL_DICT["randomforest"]

    # First, call the method_definition_function with the classifier/regressor boolean defined...
    ml_model_data = method_definition_function(classifier=False)

    # Now we instantiate the model from the ml_model_data.model attribute...
    ml_model = ml_model_data.model

    # ... then train our model...
    trained_model = experiment_obj.one_shot_train(
        ml_model, processed_datasets, parameters={"random_state": args.model_seed}
    )

    # Get the predictions and evaluate the performance.
    predictions = experiment_obj.predict(processed_datasets, trained_model)

    # Add our custom metric
    experiment_obj.add_metric(ave_abs_pct_error, "avg_abs_pct_error")

    print("Evaluation against predictions:")
    results = experiment_obj.evaluate_predictions(
        processed_datasets.test_data.labels,
        predictions=predictions,
    )

    print("\n\nEvaluation against a baseline train_data label mean:")
    results = experiment_obj.evaluate_predictions(
        processed_datasets.test_data.labels,
        predictions=predictions,
        baseline_value=processed_datasets.train_data.labels.mean(),
    )
