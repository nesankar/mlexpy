import pandas as pd
from pathlib import Path
from typing import Union, Callable, Optional
from mlexpy import processor


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
