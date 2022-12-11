import sys
from pathlib import Path
import numpy as np
from typing import List
import argparse
from sklearn.datasets import load_iris

sys.path.append(str(Path.cwd()))
print(sys.path)
from mlexpy import pipeline_utils, experiment

from from_module_example import IrisPipeline
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

    parser.add_argument(
        "-m",
        "--model_type",
        help="What is the model  to use?",
        choices=["random_forest", "sgdclassifier"],
        type=str,
    )

    return parser.parse_args(args)


"""An example similar to the notebook previously, here with more complexity shown as a script, and CL runnable, and showing
hyperparameter tuning via cv search.

One possible pythonic way to store our model definitions is via the MlModelInfo named tuple, with
examples shown in the model_defs.py file. This proivdes us one source of truth to store where we define the information about
any given machine learning model (following the fairly standardized sklearn / model and hyperparameter frameworks).
Usage is shown below.
"""

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # First, set the random seed(s) for the experiment

    model_rs = np.random.RandomState(args.model_seed)
    process_rs = np.random.RandomState(args.process_seed)

    # First, read in the dataset as a dataframe. Because mlexpy is meant to be an exploratory/experimental tool,
    # dataframes are preferred for their readability.
    data = load_iris(as_frame=True)
    features = data["data"]
    labels = data["target"]

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
        cv_split_count=5,
        model_tag="example_development_model",
        process_tag="example_development_process",
        model_dir=Path(__file__).parent,
    )

    experiment_obj.set_pipeline(IrisPipeline)

    # Now begin the experimentation, start with performing the data processing...
    processed_datasets = experiment_obj.process_data()

    # ... then train our model. But this time, define our model based on what we have written into our model dictionary
    method_definition_function = MODEL_DICT["randomforest"]

    # First, call the method_definition_function with the classifier/regressor boolean defined...
    ml_model_data = method_definition_function(classifier=True)

    # Now we instantiate the model from the ml_model_data.model attribute...
    ml_model = ml_model_data.model(random_state=model_rs)

    # ... then train our model. This time pass the ml_model_data hyperparameters to search over (this is a dictionary).
    # By default, if the params are passed, the a form of cross validated search will be performed over the provided
    # parameters. Unless defined otherwise, random search over the hyperparameter space is performed.
    trained_model = experiment_obj.train_model(
        ml_model,
        processed_datasets,
        params=ml_model_data.hyperparams,  # If this is passed, then cross validation search is performed, but slow.
        cv_model="random_search",
        cv_iterations=20,
    )

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

    # ... also evaluate the ROC based metrics
    roc_results = experiment_obj.evaluate_roc_metrics(
        processed_datasets, class_probabilities, trained_model
    )
