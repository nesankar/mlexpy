import sys
import pandas as pd
from pathlib import Path
import numpy as np
from typing import List
import argparse

sys.path.append(str(Path.cwd()))
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
        default=1,
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


"""An example similar to the notebook previously, here with more complexity shown as a script, and CL runnable.
In this case show how to load a historically trained model to evaluate a new dataset.
"""

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # First, set the random seed(s) for the experiment

    model_rs = np.random.RandomState(args.model_seed)
    process_rs = np.random.RandomState(args.process_seed)

    # Now, this time, load from a file... Here use the "NEW" dataset (the other 1/3 we created in script 2)
    data = pd.read_csv(Path(__file__).parent / "data" / "new_data.csv")
    target_col = "target"
    labels = data[target_col]
    features = data[[col for col in data.columns if col != target_col]]

    # In this case, all we are going to be doing is evaluating our "new" data. We can do this by setting the test frac to 1

    experiment_setup = pipeline_utils.get_stratified_train_test_data(
        train_data=features,
        label_data=labels,
        test_frac=args.test_frac,
        random_state=process_rs,
    )

    # ... here, instantiate the experiment class...
    experiment_obj = experiment.ClassifierExperiment(
        train_setup=experiment_setup.train_data,
        test_setup=experiment_setup.test_data,
        model_tag="example_stored_model",
        process_tag="example_stored_process",
        model_dir=Path(
            __file__
        ).parent,  # This means I will look for the models alongside this script
    )

    experiment_obj.set_pipeline(IrisPipeline)

    # Now begin the experimentation, start with performing the data processing...
    processed_datasets = experiment_obj.process_data_from_stored_models()

    # ... then LOAD our model...
    loaded_model = experiment_obj.load_model()

    # Get the predictions and evaluate the performance.
    predictions = experiment_obj.predict(processed_datasets, loaded_model)
    class_probabilities = experiment_obj.predict(
        processed_datasets, loaded_model, proba=True
    )
    results = experiment_obj.evaluate_predictions(
        processed_datasets.test_data.labels,
        predictions=predictions,
        class_probabilities=class_probabilities,
    )
