import sys
from pathlib import Path
import numpy as np
from typing import List
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

sys.path.append(str(Path.cwd()))
print(sys.path)
from mlexpy import pipeline_utils, utils, experiment

from from_module_example import IrisPipeline


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
        choices=["random_forest", "sgdclassifier"],
        type=str,
    )

    return parser.parse_args(args)


"""An example similar to the notebook previously, here with more complexity shown as a script, CL runnable, with an initial datafiltering task, and using binary classification."""

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # First, set the random seed(s) for the experiment

    model_rs = np.random.RandomState(args.model_seed)
    process_rs = np.random.RandomState(args.process_seed)

    # First, read in the dataset as a dataframe. Because mlexpy is meant to be an exploratory/experimental tool,
    # dataframes are preferred for their readability.
    data = load_iris(as_frame=True)
    data = data["frame"]

    # Image that we only want to keep data that is 0 or 1 target types. We can use the utils.initial_filtering() function and
    # and a column filter dict:

    # First, define the filter function, to return True for records we would like to keep...
    def target_col_filter(input: int) -> bool:
        """Return a True for 1 or 0 values"""
        if input == 1 or input == 0:
            return True
        else:
            return False

    # ... Now create a dict, with the key for the column we want to apply this function too. The value must be a list of functions...
    filter_fn_dict = {"target": [target_col_filter]}

    # ... and lastly, now pass this and the data to initial_filtering()
    data = utils.initial_filtering(data, filter_fn_dict)

    target_col = "target"
    labels = data[target_col]
    features = data[[col for col in data.columns if col != target_col]]

    print(f"As we can see, now the data only has 2 target values: {labels.unique()}.")
    # Now, generate the ExperimentSetup object, that splits the dataset for training and testing.
    experiment_setup = pipeline_utils.get_stratified_train_test_data(
        train_data=features,
        label_data=labels,
        test_frac=args.test_frac,
        random_state=process_rs,
    )

    # This provides us with a named tuple, with attributes of .train_data and .test_data
    # each one with attributes of .obs and .labels. For example...
    train_label_count = experiment_setup.train_data.labels.shape[0]
    test_label_count = experiment_setup.test_data.labels.shape[0]
    total_data_count = features.shape[0]

    print(
        f"Train labels are {round((total_data_count - train_label_count) / total_data_count * 100, 2)}% of the original data ({train_label_count})."
    )
    print(
        f"Test labels are {round((total_data_count - test_label_count) / total_data_count * 100, 2)}% of the original data ({test_label_count})."
    )

    # Define the experiment
    experiment_obj = experiment.ClassifierExperiment(
        train_setup=experiment_setup.train_data,
        test_setup=experiment_setup.test_data,
        cv_split_count=20,
        model_tag="example_development_model",
        process_tag="example_development_process",
        model_dir=Path(__file__).parent,
    )

    experiment_obj.set_pipeline(IrisPipeline)

    # Now begin the experimentation, start with performing the data processing...
    processed_datasets = experiment_obj.process_data()

    # ... then train our model...
    trained_model = experiment_obj.train_model(
        RandomForestClassifier(
            random_state=model_rs
        ),  # This is why we have 2 different random states...
        processed_datasets,
        # model_algorithm.hyperparams,  # If this is passed, then cross validation search is performed, but slow.
    )

    # Get the predictions and evaluate the performance.
    # Get the predictions and evaluate the performance.
    predictions = experiment_obj.predict(processed_datasets, trained_model)
    class_probabilities = experiment_obj.predict(
        processed_datasets, trained_model, proba=True
    )
    results = experiment_obj.evaluate_predictions(
        processed_datasets.test_data.labels,
        predictions=predictions,
        class_probabilities=class_probabilities,
    )

    # Because this is a binary classification case, we can easily perform ROC curve evaluations:
    experiment_obj.evaluate_roc_metrics(
        processed_datasets, class_probabilities, trained_model
    )
