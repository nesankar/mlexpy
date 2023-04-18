import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List
import argparse
from sklearn.ensemble import RandomForestClassifier
from mlexpy import pipeline_utils, experiment

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


"""Similar to the notebook previously, here showing how to use the model dumping and loading features."""

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # First, set the random seed(s) for the experiment

    model_rs = np.random.RandomState(args.model_seed)
    process_rs = np.random.RandomState(args.process_seed)

    # Now, this time, load from a file... (This is just 2/3 of the iris data)
    data = pd.read_csv(Path(__file__).parent / "data" / "original_data.csv")
    data = data.drop(columns="Unnamed: 0")
    target_col = "target"
    labels = data[target_col]
    features = data[[col for col in data.columns if col != target_col]]

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
        model_tag="example_stored_model",
        process_tag="example_stored_process",
        model_dir=Path(
            __file__
        ).parent,  # This means I will place the models alongside this script
    )

    # Set the pipeline attribute
    experiment_obj.set_pipeline(IrisPipeline)

    # Now begin the experimentation, start with performing the data processing...
    processed_datasets = experiment_obj.process_data()

    # ... then train our model...
    trained_model = experiment_obj.one_shot_train(
        RandomForestClassifier,
        processed_datasets,
        parameters={"random_state": args.model_seed},
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

    # Here we store the model
    experiment_obj.store_model(trained_model)
    # Now, test that the results are the same if loading from a disk.

    # Start with performing the data processing...
    loaded_datasets = experiment_obj.process_data_from_stored_models()

    # ... then LOAD our model...
    loaded_model = experiment_obj.load_model()

    # Get the predictions and evaluate the performance.
    loaded_predictions = experiment_obj.predict(loaded_datasets, loaded_model)
    loaded_class_probabilities = experiment_obj.predict(
        loaded_datasets, loaded_model, proba=True
    )
    loaded_results = experiment_obj.evaluate_predictions(
        loaded_datasets.test_data.labels,
        predictions=loaded_predictions,
        class_probabilities=loaded_class_probabilities,
    )

    for metric, result in results.items():
        print(
            f"""From script: {metric}: {result} -- From loading {metric}: {loaded_results[metric]}\n"""
        )
