import pandas as pd
from pathlib import Path
from typing import Union, Callable, Optional
from mlexpy import processor, experiment, pipeline_utils


class IrisPipeline(processor.ProcessPipelineBase):
    def __init__(
        self,
        process_tag: str = "example_development_process",
        model_dir: Optional[Union[str, Path]] = None,
        model_storage_function: Optional[Callable] = None,
        model_loading_function: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            process_tag, model_dir, model_storage_function, model_loading_function
        )

    # Now -- define the .process_data() method.
    def process_data(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """All data processing that is to be performed for the iris classification task."""

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
        else:
            # Here we can ONLY apply the transformation
            model_features = self.transform_model_based_features(df)

        # Imagine we only want to use the scaled features for prediction, then we retrieve only the scaled columns.
        # (This is easy because the columns are renamed with the model name in the column name)
        prediction_df = model_features[
            [col for col in model_features if "standardscaler" in col]
        ]

        return prediction_df

    # For Example (4) -- create an alternative process method.
    def process_data_keep_all_columns(
        self, df: pd.DataFrame, training: bool = True
    ) -> pd.DataFrame:
        """All data processing that is to be performed for the iris classification task."""

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
        else:
            # Here we can ONLY apply the transformation
            model_features = self.transform_model_based_features(df)

        all_feature_df = model_features

        # --- Part removed for illustrative example -----
        # # Imagine we only want to use the scaled features for prediction, then we retrieve only the scaled columns.
        # # (This is easy because the columns are renamed with the model name in the column name)
        # prediction_df = all_feature_df[
        #     [col for col in all_feature_df if "standardscaler" in col]
        # ]

        return all_feature_df

    def fit_model_based_features(self, df: pd.DataFrame) -> None:
        # Here we do any processing of columns that will require a model based transformation / engineering.

        # In this case, simply fit a standard (normalization) scaler to the numerical columns.
        # This case will result in additional columns on the dataframe named as
        # "<original-column-name>_StandardScaler()".

        # Note: there are no returned values for this method, the result is an update in the self.column_transformations dictionary

        for column in df.columns:
            if not self.check_numeric_column(df[column]):
                continue
            self.fit_scaler(df[column], standard_scaling=True)

    def fit_model_based_features_pca(self, df: pd.DataFrame, drop_columns=True) -> None:
        # Here we do any processing of columns that will require a model based transformation / engineering.

        # In this case, simply fit a standard (normalization) scaler to the numerical columns.
        # This case will result in additional columns on the dataframe named as
        # "<original-column-name>_StandardScaler()".

        # Note: there are no returned values for this method, the result is an update in the self.column_transformations dictionary

        self.fit_pca(df, n_components=2, drop_columns=drop_columns)


class IrisExperiment(experiment.ClassifierExperimentBase):
    def __init__(
        self,
        train_setup: pipeline_utils.MLSetup,
        test_setup: pipeline_utils.MLSetup,
        cv_split_count: int = 5,
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
        self,
        process_method_str: str = "process_data",
        from_file: bool = False,
    ) -> pipeline_utils.ExperimentSetup:

        # Now get the the data processing method defined in process_method_str.
        process_method = getattr(self.pipeline, process_method_str)

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
