import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import List, Any, Union, Optional, Callable, Set
from joblib import dump, load
import sys
from pathlib import Path
from glob import glob
import logging
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from mlexpy.utils import df_assertion, series_assertion, make_directory
from mlexpy.defaultordereddict import DefaultOrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FeatureReducer:
    """
    Simple class to use to be able to store a feature dropping model with all alternative models.

    Attributes
    ----------
    columns_to_drop : Set[str]
        A list of the columns that should be dropped form the dataset.

    Methods
    -------
    fit(self, columns_to_drop: Set[str])
       Define the columns that we would like to drop.
    transform(self, df: pd.DataFrame)
        Return the provided dataframe with only the columns remaining that are NOT in the columns_to_drop attribute
    """

    def __init__(self) -> None:
        self.columns_to_drop: Optional[Set[str]] = None

    def fit(self, columns_to_drop: Set[str]) -> None:
        """Set the class attribute as the provided columns_to_drop argument.

        Parameters
        ----------
        columns_to_drop : List[str]
            A list of the columns we want to drop from a dataframe.

        Returns
        -------
        None
        """
        self.columns_to_drop = columns_to_drop

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the columns stored as the columns_to_drop attribute.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame we want to drop columns from that are present in the columns_to_drop attribute.

        Returns
        -------
        pd.DataFrame
        """
        if (
            self.columns_to_drop is None
        ):  # cant be just "not" logic b/c empty lists return True from a "not" check
            raise ValueError(
                "The .columns_to_drop attribute has not been set yet. Make sure to run .fit(<list-of-cols-to-drop>) before .transform()"
            )
        columns_to_keep = [col for col in df.columns if col not in self.columns_to_drop]
        return df[columns_to_keep]


class ProcessPipelineBase:
    """
    Base data processing class.


    Attributes
    ----------
    process_tag : string
        The tag to use when storing process relevant models. (ex. column transformers)

    model_dir : Union[string, Path]
        The path to write to to store all models, if models are going to be stored.

    _default_label_encoder : Any
        The default tool to use to encode any labels, in a categorical setting. By default uses the sklearn label encoder.

    data_transformations : DefaultOrderedDict
        The ordered dictionary used to store any dataset based feature transformation that requires a model. For example a PCA model, or standard scaling to a column.
        We need to fit this PCS model ONLY on the train data, and then store the PCA model in order to apply it to the testing dataset.

    store_models : bool
        A boolean to designate if models should be stored or not.

    columns_to_drop : Set[str]
        A list that stores columns you would like to drop. By default, all columns are kept. Alternatively, columns are
        added to this list if desired when performing the model fit.


    Methods
    -------
    make_storage_dir()
       Used to create a directory at the class attribute model_dir.
    check_numeric_column(col: pd.Series)
        Check if a pandas Series is numeric, return True if so.
    fit_check(model: Any)
        Check if a model has a fit method -- if so raise a warning that it should be saved.
    fit_label_encoder(self, all_labels: pd.Series)
        Fit a label encoder to the provided all_labels data.
    encode_labels(self, labels: pd.Series)
        Do the actual label encoding.
    default_store_model(self, model: Any, model_tag: str)
        Store a model using the default tooling. This will be a jobilib dump, or the xgboost native store_model capability.
    default_load_model(self, model_tag: str, model: Optional[Any] = None)
        Load a model using the default tooling. This will be a joblib load, or a model's native load tooling. Ex. such as in xgboost models.
    process_data(self, df: pd.DataFrame, label_series: pd.Series, training: bool = True)
        The method to do all data processing. NOTE: This must be overwritten in the child class inheriting this class.
    drop_columns(self, df: pd.DataFrame)
        Drop the columns stored in the columns to drop list.
    keep_columns(df: pd.DataFrame, keep_cols: List[str])
        Drop all columns NOT in the passed Keep_cols list.
    set_default_encoder(self, encoder: Any)
        Set the default label encoder to be any encoder class desired.
    get_best_cols(self, df: pd.DataFrame, labels: pd.Series, col_count: Optional[int] = None)
        Perform sklearn's best column selection.
    fit_data_model(self, model: Any, feature_data: Union[pd.DataFrame, pd.Series], fit_method_name: str = "fit")
        Perform the data fitting and respective model storage.
    fit_scaler(self, feature_data: pd.Series, standard_scaling: bool = True)
        Fit a scaler model to a column, options include a standard scaler or a min-max scaler
    fit_model_based_features(self, df: pd.DataFrame)
        Fit the models based features to the passed dataframe. NOTE: This must be overwritten in the child class inheriting this class.
    transform_model_based_features(self, df: pd.DataFrame)
        Perform all transformations of columns according to the entries in teh column_transformation dict, or according to what can be loaded at the provided model_dir.
    dump_feature_based_models()
        Store all of the current models performing some feature transformation to disk.
    load_feature_based_models()
        Load all of the column transformation models that can be found at the model_dir path.

    Properties
    ----------
    label_encoder : LabelEncoder
        Load the default label encoder, but define it only when initially called upon.
    """

    def __init__(
        self,
        process_tag: str = "_development",
        model_dir: Optional[Union[str, Path]] = None,
        model_storage_function: Optional[Callable] = None,
        model_loading_function: Optional[Callable] = None,
        store_models: bool = True,
        rand_int: int = 10,
    ) -> None:

        self.process_tag = process_tag
        self.model_dir: Path = Path()
        self.k_worst_cols: List[str] = []
        self._default_label_encoder = LabelEncoder
        self.data_transformations = DefaultOrderedDict(lambda: [])
        self.store_models = store_models
        self.columns_to_drop: Set[str] = set()
        self.feature_reducer = FeatureReducer()
        self.rand_int = rand_int

        # Set up any model IO
        if not model_storage_function:
            logger.info(
                "No model storage function provided. Using the default class method (joblib, or .store_model native method)."
            )
            self.store_model = self.default_store_model
        else:
            logger.info(f"Set the model storage function as: {model_storage_function}")
            self.store_model = model_storage_function
        if not model_loading_function:
            logger.info(
                "No model loading function provided. Using the default class method (joblib, or .load_model native method)."
            )
            self.load_model = self.default_load_model
        else:
            logger.info(f"Set the model loading function as: {model_loading_function}")
            self.load_model = model_loading_function

        if not model_dir:
            logger.info(
                f"No model location provided. Creating a .models/ at: {sys.path[-1]}"
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

    def make_storage_dir(self) -> None:
        """Create the directory of the mode_dir class attribute.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not self.model_dir.is_dir():
            make_directory(self.model_dir)

    @staticmethod
    def check_numeric_column(col: pd.Series) -> bool:
        """Simply check if a column is numeric.

        Parameters
        ----------
        col : pd.Series
            The data to check, as a pandas Series.

        Returns
        -------
        None
        """
        return is_numeric_dtype(col)

    @staticmethod
    def fit_check(model: Any) -> None:
        """Check if a model has a .fit() method, and if so lof that the model exists and that it should be stored to file.

        Parameters
        ----------
        model : Any
            A model that is checked for a fit method. Note, this needs to be a class.

        Returns
        -------
        None
        """
        if hasattr(model, "fit"):
            logger.warning(
                f"The provided {model} model has a .fit() method. Make sure to store the resulting fit method for proper train test separation."
            )

    def set_default_encoder(self, encoder: Any) -> None:
        """Set the desired label encoder.

        Parameters
        ----------
        encoder : Any
            The encoder that is to be set as the current default encoder.

        Returns
        -------
        None
        """
        self._default_label_encoder = encoder

    @property
    def label_encoder(self) -> Any:
        if not hasattr(self, "_label_encoder"):
            logging.info(f"Setting the label encoder as {self._default_label_encoder}.")
            self._label_encoder = self._default_label_encoder()
        return self._label_encoder

    def fit_label_encoder(self, all_labels: pd.Series) -> None:
        """Fit the designated label encoder stored in the .label_encoder() property.

        Parameters
        ----------
        all_labels : pd.Series
            The data that is going to be encoded, stored as a Pandas Series.

        Returns
        -------
        None
        """
        series_assertion(all_labels)
        self.label_encoder.fit(all_labels.values)
        return None

    def encode_labels(self, labels: pd.Series) -> pd.Series:
        """Perform the actual label encoding.

        Parameters
        ----------
        all_labels : pd.Series
            The data that is going to be encoded, stored as a Pandas Series.

        Returns
        -------
        pd.Series
        """
        series_assertion(labels)
        try:
            coded_labels = self.label_encoder.transform(labels.values)
        except NotFittedError:
            # Need to first do the encoder fitting
            logger.warn(
                "The label encoder has not been fit. The current encoder is being fit on the TRAINING data."
            )
            self.fit_label_encoder(labels)
            coded_labels = self.label_encoder.transform(labels.values)
        return pd.Series(coded_labels, name=labels.name, index=labels.index)

    def default_store_model(self, model: Any, model_path: Union[str, Path]) -> None:
        """Given a calculated model, store it locally using joblib.

        Longer term/other considerations can be found here: https://scikit-learn.org/stable/model_persistence.html

        Parameters
        ----------
        model : Any
            The model that is to be stored

        model_tag : str
            The string tag to associate with this specific model.

        Returns
        -------
        None
        """
        self.make_storage_dir()

        if hasattr(model, "save_model"):
            # use the model's saving utilities, specifically beneficial wish xgboost. Can be beneficial here to use a json
            logger.info(f"Found a save_model method in {model}")
            model.save_model(f"{model_path}.mdl")
        else:
            logger.info(f"Saving the {model} model using joblib.")
            dump(model, f"{model_path}.joblib")
        logger.info(f"Dumped {model} to: {model_path}")

    def default_load_model(
        self, model_path: Union[str, Path], model: Optional[Any] = None
    ) -> Any:
        """Given a model name, load it from storage

        Longer term/other considerations can be found here: https://scikit-learn.org/stable/model_persistence.html

        Parameters
        ----------
        model_tag : str
            The string tag to associate with this specific model.

        model : Optional[Any]
            The model to load. This is optional for a case where a model is instantiated, and then loaded from file in to the provided model instance.

        Returns
        -------
        Any (the model type loaded)
        """
        if hasattr(model, "load_model") and model:
            # use the model's loading utilities -- specifically beneficial with xgboost
            logger.info(f"Found a load_model method in {model}")
            loaded_model = model.load_model(model_path)
        else:
            logger.info(f"Loading {model} from: {model_path}")
            loaded_model = load(model_path)
        logger.info(f"Retrieved {model} from: {model_path}")
        return loaded_model

    def process_data(
        self,
        df: pd.DataFrame,
        training: bool = True,
        label_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Perform here all steps of the data processing for feature engineering

        Parameters
        ----------
        df : pd.DataFrame
            The input to-be-processed dataset.

        label_series : pd.Series
            The labels associated with the respective targets.

        training : bool
            A boolean declaring if this is training or testing dataset. For testing, we will only APPLY any transformations (not fit them).

        Returns
        -------
        pd.DataFrame

        Notes
        -----
        Example structure to define in the child class:

            def process_data(self, df: pd.DataFrame, label_series: pd.Series,  training: bool = True) -> pd.DataFrame:
                # Do a copy of the passed df
                df = df.copy()

                <DO ANY HARD-CODEABLE FEATURE ENGINEERING HERE (DOES NOT REQUIRE LEARNING ON THE TRAINING SET)>

                # Now perform the training / testing dependent feature processing. This is why a `training` boolean is passed.
                if training:
                    # Now FIT all of the model based features (ex. scaling, imputing)...
                    self.fit_model_based_features(df)
                    # ... and get the results of a transformation of all model based features.
                    model_features = self.transform_model_based_features(df)
                else:
                    # Here we can ONLY apply the transformation
                    model_features = self.transform_model_based_features(df)

                <DO ANY FINAL STEPS HERE, EX. COLUMN FILTERING, ETC.

                return model_features
        """
        raise NotImplementedError("This needs to be implemented in the child class.")

    def drop_columns(
        self, df: pd.DataFrame, cols_to_drop: Union[Set[str], List[str]]
    ) -> pd.DataFrame:
        """Simply drop the columns that are passed to the method.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe that we want to drop columns on.

        cols_to_drop : Union[Set[str], List[str]]
            The list of columns names that we want to drop.

        Returns
        -------
        pd.DataFrame
        """
        df_assertion(df)
        # First do an intersection of the df's columns and those to drop.
        dropping_cols = [col for col in df.columns if col in cols_to_drop]
        return df.drop(columns=dropping_cols)

    @staticmethod
    def keep_columns(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
        df_assertion(df)
        """Given the defined keep_cols list, drop all other columns

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe that we want to drop columns on.

        keep_cols : List[str]
            The list of columns names that we want to KEEP in the df.

        Returns
        -------
        pd.DataFrame
        """
        return df[keep_cols]

    def fit_data_model(
        self,
        model: Any,
        feature_data: Union[pd.DataFrame, pd.Series],
        drop_columns: bool = False,
        fit_method_name: str = "fit",
    ) -> None:
        """Do all model fitting here in a dataset (or subset) WIDE manner, such as dimensionality reduction.
        This guarantees that all models will be stored, dumped, and reloaded correctly, and conveniently.

        Parameters
        ----------
        model : Any
            This is the instantiated model that you would like to fit to the data. Ex. an instantiated standard scaler.

        feature_data : pd.DataFrame
            This is the DataFrame that you would like to apply the model to.

        fit_method_name : str
            This is the name of the method to use to fit the model provided to the method. By default this will just be "fit"
            but for customizeability, this can be any name.

        drop_columns : bool
            Boolean flag to define if the columns transformed should be stored and additionally be dropped.

        Returns
        -------
        None
        """
        if not hasattr(model, fit_method_name):
            raise NameError(
                f"The {model} model has no {fit_method_name} method to use to fit a model."
            )
        fit_call = getattr(model, fit_method_name)

        # Now perform the fitting according to the respective datatype provided.
        if isinstance(feature_data, pd.DataFrame):
            fit_call(feature_data.values)
            # get the feature specific key string
            key_string = "~~".join(feature_data.columns)
            if drop_columns:
                self.columns_to_drop.update(feature_data.columns)

        elif isinstance(feature_data, pd.Series):
            fit_call(feature_data.values.reshape(-1, 1))
            # get the feature specific key string
            key_string = feature_data.name
            if drop_columns:
                self.columns_to_drop.update([feature_data.name])
        else:
            logger.warning(
                f"The data to fit was not a pandas dataframe or series: {type(feature_data)}. Skipping and not applying the {model} model."
            )

        # Store the fit scaler to apply to the testing data.
        self.data_transformations[key_string].append(model)

    def fit_scaler(
        self,
        feature_data: pd.Series,
        standard_scaling: bool = True,
        drop_columns: bool = False,
        **kwargs,
    ) -> None:
        """Perform the feature scaling here. If this a prediction method, then load and fit.

        Parameters
        ----------
        feature_data : pd.Series
            The data we would like to scale provided as a pandas Series.

        standard_scaling : bool
            A boolean flag for if standard scaling should be used or not. If passed as False, then min-max scaling is used.

        Returns
        -------
        None
        """
        if standard_scaling:
            logger.info(f"Fitting a standard scaler to {feature_data.name}.")
            scaler = StandardScaler(**kwargs)
        else:
            logger.info(f"Fitting a minmax scaler to {feature_data.name}.")
            scaler = MinMaxScaler(**kwargs)

        self.fit_data_model(scaler, feature_data, drop_columns)

    def fit_one_hot_encoding(
        self, feature_data: pd.Series, drop_columns: bool = True, **kwargs
    ) -> None:
        """Perform one-hot encoding here. If this a prediction method, then load and fit.

        Parameters
        ----------
        feature_data : pd.Series
            The data we would like to transform provided as a pandas Series.

        Returns
        -------
        None
        """
        logger.info(f"Fitting a one-hot-encoder to {feature_data.name}.")
        onehoter = OneHotEncoder(handle_unknown="ignore", sparse=False, **kwargs)
        self.fit_data_model(onehoter, feature_data, drop_columns)

    def fit_pca(
        self,
        feature_data: pd.DataFrame,
        n_components: int,
        drop_columns: bool = True,
        **kwargs,
    ) -> None:
        """Perform principal component analysis. If this a prediction method, then load and fit.

        Parameters
        ----------
        feature_data : pd.DataFrame
            The data we would like to transform provided as a pandas DataFrame.

        n_components : int
            The number of principal components to reattain after PCA.

        Returns
        -------
        None
        """
        logger.info(
            f"Fitting a pca model to {feature_data.columns} to {n_components} principal components."
        )
        pca = PCA(n_components=n_components, random_state=self.rand_int, **kwargs)
        self.fit_data_model(pca, feature_data, drop_columns=drop_columns)

    def get_correlated_columns(
        self, df: pd.DataFrame, corr_threshold: float
    ) -> List[str]:
        """Return the list of the pairs of columns with correlations greater than the threshold."""

        corr_df = df.corr()
        cols = corr_df.columns

        result_list = []
        for i, col_1 in enumerate(cols):
            for col_2 in cols[i + 1 :]:
                if corr_df.loc[col_1][col_2] >= corr_threshold:
                    result_list.append((col_1, col_2))

        cols_to_drop = [col[1] for col in result_list]
        self.columns_to_drop.update(cols_to_drop)

        return cols_to_drop

    def drop_correlated_columns(
        self,
        df: pd.DataFrame,
        thresh: float,
    ) -> None:

        """Drop any of the correlated columns according to the column_coor_thresh"""
        correlated_columns = self.get_correlated_columns(df, thresh)

        logger.info(f"Dropping: {correlated_columns}")

        return self.drop_columns(df, correlated_columns)

    def fit_model_based_features(self, df: pd.DataFrame) -> None:
        """Here do all feature engineering that requires models to be fit to the data, such as, scaling, on-hot-encoding,
        PCA, imputing, etc.

        The goal is to arrange these in a manner that makes them easily reproducible, easily understandable, and persist-able.

        Each process performed here is stored in the data_transformations dictionary, with ordering with the default key a list.
        The processes in this dictionary will be passed over IN ORDER on the test df to generate the test dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The data we would like to fit models to, as a data frame.

        Returns
        -------
        None

        Notes
        -----
        Example structure to define in the child class:

            def fit_model_based_features(self, df: pd.DataFrame) -> None:
                # In this case, simply fit a standard (normalization) scaler to the NUMERICAL columns,
                # and a custom model we have developed (commented for copy pasting.)
                # This case will result in additional columns on the dataframe named as
                # "<original-column-name>_standardscaler()".

                # Note: there are no returned values for this method, the result is an update in the self.data_transformations dictionary


                # my_custom_model = CustomModel()
                for column in df.columns:
                    if self.check_numeric_column(df[column]):
                        self.fit_scaler(df[column], standard_scaling=True)
                        self.fit_model(my_custom_data, df[column])
        """
        raise NotImplementedError("This needs to be implemented in the child class.")

    def transform_model_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Here apply all model based feature engineering models, previously fit with the fit_model_based_features method.

        The goal is to simply call this method, and perform a single, ordered set of operations on a dataset to provide
        feature engineering with models with no risk of test set training, AND the ability to load a previously trained
        feature model.

        Parameters
        ----------
        df : pd.DataFrame
            The data we would like to perform the model transformation on as a data frame.

        Returns
        -------
        pd.DataFrame
        """
        result_df = pd.DataFrame(index=df.index)
        if len(self.data_transformations) == 0:
            # First, check to see if there might be any files to load
            try:
                self.load_feature_based_models()
            except FileNotFoundError:
                logger.exception(
                    f"Note: No files were found for {self.process_tag}. If these should be found, check to make sure the models were dumped correctly. (Looked at in: {self.model_dir})"
                )
                logger.info(
                    "\nContinuing with out loading any model based column transformations."
                )
                return df
        else:
            # By default, store all models when performing a transformation
            if self.store_models:
                self.dump_feature_based_models()

        for column, transformations in self.data_transformations.items():
            if "~~" in column:
                # This means we have a multi-column transformation to perform.
                column_to_use = column.split("~~")
                data_to_use = df[column_to_use].values
                name_to_use = (
                    ""  # none here because there is not 1 column being transformed.
                )

            elif column == "Unnamed: 0":
                # We never want to use this column, so skip it. This also removes it.
                continue
            elif (
                column not in df.columns
            ):  # This also skips the feature reducer feature
                logger.info(
                    f"{column} NOT found in the provided dataframe. Skipping {transformations}."
                )
                continue
            else:
                # Lastly, we have a single column to apply a transformation to.
                column_to_use = column
                data_to_use = df[column_to_use].values.reshape(-1, 1)
                name_to_use = column
            for i, transformation in enumerate(transformations):
                logger.info(f"Applying the {transformation} to {column_to_use}")
                transformed_result = transformation.transform(data_to_use)
                # Just use the name of the transformation (assuming it is a class)
                transformation_name = transformation.__str__().lower().split("(")[0]

                if transformed_result.shape[1] > 1:
                    # Then the resulting transformation is a matrix (ex. one hot encoding). Make it dataframe ammenable
                    if hasattr(transformation, "get_feature_names_out"):
                        columns = transformation.get_feature_names_out()
                    else:
                        columns = list(range(len(transformed_result.shape[1])))
                    transformed_df = pd.DataFrame(
                        transformed_result,
                        columns=columns,
                        index=df.index,
                    )
                    transformed_df.columns = [
                        f"{name_to_use}_{transformation_name}_{col}"
                        for col in transformed_df.columns
                    ]
                    result_df = pd.concat([result_df, transformed_df], axis=1)
                else:
                    result_df[
                        f"{name_to_use}_{transformation_name}"
                    ] = transformed_result

        if not self.feature_reducer.columns_to_drop:
            self.feature_reducer.fit(self.columns_to_drop)

        return self.feature_reducer.transform(pd.concat([df, result_df], axis=1))

    def dump_feature_based_models(self) -> None:
        """Given the ordered dict of the model based features, dump each model, with the name of the model in the column_transformation dict.

        Use a process/indexed-column_name/indexed-model structure in-order to maintain the ordering.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # By default, dump the models to storage.
        self.make_storage_dir()
        # Iterate through the columns_transformer dict, storing each model and column. Use a method wide indexer
        idx = 0
        for column, transformations in self.data_transformations.items():
            column_process_model_dir = self.model_dir / column
            if not column_process_model_dir.is_dir():
                make_directory(column_process_model_dir)
            logger.info(
                f"Dumping {len(transformations)} models to {column_process_model_dir}"
            )
            for transformation in transformations:
                transformation_name = transformation.__str__().lower()
                self.store_model(
                    transformation,
                    column_process_model_dir / f"{idx}_{transformation_name}",
                )
                idx += 1

        # Lastly, store any info about the columns that should be dropped
        self.feature_reducer.fit(self.columns_to_drop)
        feature_reducer_path = self.model_dir / "feature_reducer"
        if not feature_reducer_path.is_dir():
            make_directory(feature_reducer_path)
        self.store_model(
            self.feature_reducer,
            feature_reducer_path / f"{idx}_featurereducer",
        )

    def load_feature_based_models(self) -> None:
        """Given the process tag used to instantiate the class, load all models used for feature generation.

        Parameters
        ----------
        None

        Returns
        -------
        DefaultOrderedDict
        """
        # First, get all files
        all_files = glob(f"{self.model_dir}/**/**")

        # Next extract the ordering information from each model file name
        file_ordering = []
        for file in all_files:
            file_pairing = [file, file.split("/")[-1].split("_")[0]]
            file_ordering.append(file_pairing)

        # Now, sort this list by the index value (the second item in each sub-list)
        file_ordering.sort(key=lambda entry: entry[1])

        # Finally, iterate through this file ordering, load the models and store correctly in the data_transformations_dict
        for file_pair in file_ordering:
            file = file_pair[0]
            column_name = file.split("/")[-2]
            model = self.load_model(file)
            if column_name == "feature_reducer":
                self.feature_reducer = model
            self.data_transformations[column_name].append(model)
