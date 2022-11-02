import pandas as pd
from typing import List, Any, Union, Optional, Callable
from joblib import dump, load
import sys
from pathlib import Path
import logging
from sklearn.preprocessing import OrdinalEncoder
from mlexpy.utils import df_assertion, series_assertion, make_directory

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
        self._default_label_encoder = OrdinalEncoder
        self.dataframe_assertion = df_assertion
        self.series_assertion = series_assertion

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

    @classmethod
    def prediction_call(
        cls,
        record: Union[pd.DataFrame, pd.Series],
        process_tag: str = "_production",
    ) -> pd.DataFrame:
        raise NotImplementedError("This needs to be implemented in the child class.")

    @property
    def label_encoder(self) -> Any:
        if not hasattr(self, "_label_encoder"):
            logging.info(f"Setting the label encoder as {self._default_label_encoder}.")
            self._label_encoder = self._default_label_encoder()
        return self._label_encoder

    def fit_label_encoder(self, all_labels: pd.Series) -> None:
        self.series_assertion(all_labels)
        self.label_encoder.fit(all_labels)
        return None

    def encode_labels(self, labels: pd.Series) -> pd.Series:
        self.series_assertion(labels)
        return self.label_encoder.transform(labels)

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

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
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
