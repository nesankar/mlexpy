import pandas as pd
from typing import List, Any, Union, Optional, Callable
from joblib import dump, load
import sys
from pathlib import Path
from glob import glob
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from mlexpy.utils import df_assertion, series_assertion, make_directory
from src.defaultordereddict import DefaultOrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ProcessPipelineBase:
    def __init__(
        self,
        process_tag: str = "_development",
        model_dir: Optional[Union[str, Path]] = None,
        model_storage_function: Optional[Callable] = None,
        model_loading_function: Optional[Callable] = None,
    ) -> None:
        """Instanciate the data processing pipeline. Note if for_training is True, then all models used are trained and stored, otherwise they are
        loaded from file using the process tag.

        Note: No lanugage is provided here, so functionality is dependent on being inherited into a child class
        """

        self.process_tag = process_tag
        self.model_dir: Path = Path()
        self.columns_to_drop: List[str] = []
        self._default_label_encoder = LabelEncoder
        self.dataframe_assertion = df_assertion
        self.series_assertion = series_assertion
        self.column_transformations = DefaultOrderedDict(lambda: [])

        # Set up any model IO
        if not model_storage_function:
            logger.info(
                "No model storage function provided. Using the default class method (joblib, or .store_model native method)."
            )
            self.store_model = self.default_store_model
        else:
            logger.info(f"Set the model storage funtion as: {model_storage_function}")
            self.store_model = model_storage_function
        if not model_loading_function:
            logger.info(
                "No model loading function provided. Using the default class method (joblib, or .load_model native method)."
            )
            self.load_model = self.default_load_model
        else:
            logger.info(f"Set the model loading funtion as: {model_loading_function}")
            self.store_model = model_loading_function

        if not model_dir:
            logger.info(
                f"No model location provided. Creading a .models/ at: {sys.path[-1]}"
            )
            self.model_dir = Path(sys.path[-1]) / ".models" / self.process_tag
        elif isinstance(model_dir, str):
            logger.info(
                f"setting the model path to {model_dir}. (Converting from string to pathlib.Path)"
            )
            self.model_dir = Path(model_dir) / self.process_tag
        else:
            logger.info(
                f"setting the model path to {model_dir}. (Converting from string to pathlib.Path)"
            )
            self.model_dir = model_dir / self.process_tag
        if not self.model_dir.is_dir():
            make_directory(self.model_dir)

    @staticmethod
    def fit_check(model: Any) -> None:
        if hasattr(model, "fit"):
            logger.warn(
                f"The provided {model} model has a .fit() method. Make sure to store the resulting fit method for proper train test seperation."
            )

    @property
    def label_encoder(self) -> Any:
        if not hasattr(self, "_label_encoder"):
            logging.info(f"Setting the label encoder as {self._default_label_encoder}.")
            self._label_encoder = self._default_label_encoder()
        return self._label_encoder

    def fit_label_encoder(self, all_labels: pd.Series) -> None:
        self.series_assertion(all_labels)
        self.label_encoder.fit(all_labels.values)
        return None

    def encode_labels(self, labels: pd.Series) -> pd.Series:
        self.series_assertion(labels)
        coded_labels = self.label_encoder.transform(labels.values)
        return pd.Series(coded_labels, name=labels.name)

    def default_store_model(self, model: Any, model_tag: str) -> None:
        """Given a calculated model, store it locally using joblib.
        Longer term/other considerations can be found here: https://scikit-learn.org/stable/model_persistence.html
        """
        if hasattr(model, "save_model"):
            # use the model's saving utilities, specifically beneficial wish xgboost. Can be beneficial here to use a json
            logger.info(f"Found a save_model method in {model}")
            model_path = self.model_dir / f"{model_tag}_{self.process_tag}.json"
            model.save_model(model_path)
        else:
            logger.info(f"Saving the {model} model using joblib.")
            model_path = self.model_dir / f"{model_tag}_{self.process_tag}.joblib"
            dump(model, model_path)
        logger.info(f"Dumped {model_tag} to: {model_path}")

    def default_load_model(self, model_tag: str, model: Optional[Any] = None) -> Any:
        """Given a model name, load it from storage."""

        if hasattr(model, "load_model") and model:
            # use the model's loading utilities -- specifically beneficial with xgboost
            logger.info(f"Found a load_model method in {model}")
            model_path = self.model_dir / f"{model_tag}_{self.process_tag}.json"
            loaded_model = model.load_model(model_path)
        else:
            model_path = self.model_dir / f"{model_tag}_{self.process_tag}.joblib"
            logger.info(f"Loading {model_tag} from: {model_path}")
            loaded_model = load(model_path)
        logger.info(f"Retrieved {model_tag} from: {model_path}")
        return loaded_model

    def process_data(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """Perform here all steps of the data processing for feature engineering."""
        raise NotImplementedError("This needs to be implemented in the child class.")

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simply drop the columns that have been cashed in the class."""
        self.dataframe_assertion(df)
        # First do an intersection of the df's columns and those to drop.
        droping_cols = [col for col in df.columns if col in self.columns_to_drop]
        return df.drop(columns=droping_cols)

    def keep_columns(self, df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
        self.dataframe_assertion(df)
        """Given the defined keep_cols list, drop all other columns"""
        return df[[keep_cols]]

    def set_default_encoder(self, encoder: Any) -> None:
        """Set the desired label encoder."""
        self._default_label_encoder = encoder

    def get_best_cols(
        self, df: pd.DataFrame, labels: pd.Series, col_count: Optional[int] = None
    ) -> None:
        """Compute the most informative columns, and then cache the rest in the to drop columns."""

        # If no col_count, default to the best 50%
        if not col_count:
            total_columns = len(df.columns)
            col_count = min(total_columns, total_columns // 2)

        selector = SelectKBest(k=col_count)
        selector = selector.fit(df, labels)

        self.columns_to_drop.extend(
            [col for col in df.columns if col not in selector.get_feature_names_out()]
        )

    def fit_scaler(
        self, feature_data: pd.Series, standard_scaling: bool = True
    ) -> pd.Series:
        """Perform the feature scaling here. If this a prediction method, then load and fit."""

        if standard_scaling:
            logger.info(f"Fitting a standard scaler to {feature_data.name}.")
            scaler = StandardScaler()
        else:
            logger.info(f"Fitting a minmax scaler to {feature_data.name}.")
            scaler = MinMaxScaler()

        scaler.fit(feature_data.values.reshape(-1, 1))
        # Store the fit scaler to apply to the testing data.
        self.column_transformations[feature_data.name].append(scaler)

    def fit_model_based_features(self, df: pd.DataFrame) -> None:
        """Here do all feature engineering that requires models to be fit to the data, such as, scaling, on-hot-encoding,
        PCA, etc.

        The goal is to arange these in a manner that makes them easily reproduceable, easily understandable, and persistable.

        Each process performed here is stored in the column_transofrmations dictionary, with ordering with the default key a list.
        The processes in this dictionary will be passed over IN ORDER on the test df to generate the test dataset.
        """
        raise NotImplementedError("This needs to be implemented in the child class.")

    def transform_model_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Here apply all model based feature enginnering models, previously fit with the fit_model_based_features method.

        The goal is to simply call this method, and perform a single, ordered set of operations on a dataset to provide
        feature engineering with models and no risk of test set training, AND the ability to load a previously trained
        feature model.
        """

        result_df = pd.DataFrame(index=df.index)

        for column, transformations in self.column_transformations.items():
            if column not in df.columns:
                logger.info(
                    f"{column} NOT found in the provided dataframe. Skipping {transformations}."
                )
                continue
            for i, transformation in enumerate(transformations):
                logger.info(f"Applying the {transformation} to {column}")
                result_df[
                    f"{column}_{transformation.__str__().lower()}"
                ] = transformation.transform(df[column].values.reshape(-1, 1))

        return pd.concat([df, result_df], axis=1)

    def dump_feature_based_models(self) -> None:
        """Given the ordered dict of the model based features, dump each model, with the name of the model in the column_transormations_dict.

        Use a process/indexed-column_name/indexed-model strucutre inorder to maintain the ordering.
        """

        # Iterate through the columns_transformer dict, storing each model and column. Use a method wide indexer
        idx = 0
        for column, transformations in self.column_transformations.items():
            column_process_model_dir = self.model_dir / column
            if not column_process_model_dir.is_dir():
                make_directory(column_process_model_dir)
            logger.info(
                f"Dumping {len(transformations)} models to {column_process_model_dir}"
            )
            for transformation in transformations:
                transformation_name = transformation.__str__().lower()
                dump(
                    transformation,
                    column_process_model_dir / f"{idx}_{transformation_name}.joblib",
                )
                idx += 1

    def load_feature_based_models(self) -> DefaultOrderedDict[str, List[Any]]:
        """Given the process tag used to instanciate the class, load all models used for feature generation."""

        # First, get all files
        all_files = glob(f"{self.model_dir}/**/**")

        # Next extract the ordering information from each model file name
        file_ordering = []
        for file in all_files:
            file_pairing = [file, file.split("/")[-1].split("_")[0]]
            file_ordering.append(file_pairing)

        # Now, sort this list by the index value (the second item in each sub-list)
        file_ordering.sort(key=lambda entry: entry[1])

        # Finally, iterate through this file ordering, load the models and store correctly in the column_transformations_dict
        for file_pair in file_ordering:
            file = file_pair[0]
            column_name = file.split("/")[-2]
            model = load(file)
            self.column_transformations[column_name].append(model)
