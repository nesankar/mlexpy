import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Callable
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

sys.path.append(str(Path.cwd()))
print(sys.path)
from mlexpy import experiment, pipeline_utils, processor


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


"""Similar to the notebook previously, here with more complexity shown as a script, and CL runnable"""
# To begin with, develop the processor:
class IrisPipeline(processor.ProcessPipelineBase):
    def __init__(
        # All of the Optional arguments are not strictly necessary but shown for brevity.
        self,
        process_tag: str = "iris_development",
        model_dir=None,
        model_storage_function=None,
        model_loading_function=None,
    ) -> None:
        super().__init__(
            process_tag, model_dir, model_storage_function, model_loading_function
        )

    # Now -- define the .process_data() method.
    def process_data(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        # Now, simply do all feature engineering in this method, and return the final data/feature set to perform
        # predictions on.

        # Imagine we have 1 desired feature to engineer, petal/sepal area, and then normalize the feature values.
        # We need to pay attention in the normalizing step, because we can ONLY apply the normalize to the test
        # set, thus we will have a fork in the process when doing the feature normalization.

        # In order to easily maintain reproducibility in data processing, any model based feature engineering (such
        # as normalization) is done by creating a specific data structure storing the order of steps for processing each column,
        # and the model that should be applied. This is somewhat similar to the ColumnTransformer in sklearn.

        # Model based features are handled in the .fit_model_based_features() method, described below.

        # Lets begin:

        # Do a copy of the passed df
        df = df.copy()

        # First, compute the petal / sepal areas (but make the columns simpler)
        df.columns = [col.replace(" ", "_").strip("_(cm)") for col in df.columns]

        for object in ["petal", "sepal"]:
            df[f"{object}_area"] = df[f"{object}_length"] * df[f"{object}_width"]

        # Now perform the training / testing dependent feature processing. This is why a `training` boolean is passed.
        if training:
            # Now FIT all of the model based features...
            self.fit_model_based_features(df)
            # ... and get the results of a transformation of all model based features.
            model_features = self.transform_model_based_features(df)

            # Here we add logic to dump all model info
            self.dump_feature_based_models()

        else:
            # Here we can ONLY apply the transformation
            model_features = self.transform_model_based_features(df)

        # Imagine we only want to use ONLY the scaled features for prediction, then we retrieve only the scaled columns.
        # (This is easy because the columns are renamed with the model name in the column name)
        prediction_df = model_features[
            [col for col in model_features if "standardscaler" in col.lower()]
        ]

        return prediction_df

    def fit_model_based_features(self, df: pd.DataFrame) -> None:
        # Here we do any processing of columns that will require a model based transformation / engineering.

        # In this case, simply fit a standard (normalization) scaler to the numerical columns.
        # This case will result in additional columns on the dataframe named as
        # "<original-column-name>_StandardScaler()".

        # Note: there are no returned values for this method, the result is an update in the self.column_transformations dictionary
        for column in df.columns:
            if df[column].dtype not in ("float", "int"):
                continue
            self.fit_scaler(df[column], standard_scaling=True)


# Next, the ClassifierExperiment
class IrisExperiment(experiment.ClassifierExperimentBase):
    def __init__(
        self,
        train_setup: pipeline_utils.MLSetup,
        test_setup: pipeline_utils.MLSetup,
        cv_split_count: int,
        rnd_int: int = 100,
        model_dir: Optional[Union[str, Path]] = None,
        model_storage_function: Optional[Callable] = None,
        model_loading_function: Optional[Callable] = None,
        model_tag: str = "example_development_model",
        process_tag: str = "example_development_process",
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
        self, process_method_str: str = "process_data"
    ) -> pipeline_utils.ExperimentSetup:

        processor = IrisPipeline(process_tag=self.process_tag, model_dir=self.model_dir)

        # Now do the data processing on the method defined in process_method_str.
        process_method = getattr(processor, process_method_str)
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


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # First, set the random seed(s) for the experiment

    model_rs = np.random.RandomState(args.model_seed)
    process_rs = np.random.RandomState(args.process_seed)

    # Now, this time, load from a file... (This is just 2/3 of the iris data)
    data = pd.read_csv(Path(__file__).parent / "data" / "original_data.csv")
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
    experiment_obj = IrisExperiment(
        train_setup=experiment_setup.train_data,
        test_setup=experiment_setup.test_data,
        cv_split_count=20,
        model_tag="example_development_model",
        process_tag="example_development_process",
        model_dir=Path(
            __file__
        ).parent,  # This means I will place the models alongside this script
    )

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
    predictions = experiment_obj.predict(processed_datasets, trained_model)
    class_probabilities = experiment_obj.predict(
        processed_datasets, trained_model, proba=True
    )
    results = experiment_obj.evaluate_predictions(
        processed_datasets,
        predictions=predictions,
        class_probabilities=class_probabilities,
    )

    # Here we store the model
    experiment_obj.store_model(trained_model)
