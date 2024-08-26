import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Callable, Union, List
from collections import namedtuple
from itertools import product

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""Some ML utility tools."""
MLSetup = namedtuple("MLSetup", ["obs", "labels"])
ExperimentSetup = namedtuple("ExperimentSetup", ["train_data", "test_data"])
CVEval = namedtuple("CVEval", ["mean", "median", "std"])


def get_stratified_train_test_data(
    train_data: pd.DataFrame,
    label_data: pd.Series,
    random_state: np.random.RandomState,
    test_frac: float = 0.3,
    stratify: bool = True,
) -> ExperimentSetup:
    """Perform a structured training and testing split of a dataset.

    Parameters
    ----------
    train_data : pd.DataFrame
        The dataframe containing our feature data.

    label_data : pd.Series
        The labels or targets associated with the respective training dataframe.

    random_state : np.random.RandomState
        The random state to use to perform the splitting.

    test_frac : float
        The fraction of data that you want to use for testing. NOTE: if 1, then only test data will be returned (at it will be the entire dataset).

    stratify : bool
        A boolean to designate if the splitting should be stratified.

    Returns
    -------
    ExperimentSetup
    """
    # First, test to see if the test frac is 1. Essentially this is to initialize a dataset ONLY for testing.
    if test_frac == 1:
        return ExperimentSetup(
            MLSetup(pd.DataFrame(), pd.Series()), MLSetup(train_data, label_data)
        )

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            train_data,
            label_data,
            test_size=test_frac,
            stratify=label_data,
            random_state=random_state,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            train_data,
            label_data,
            test_size=test_frac,
            random_state=random_state,
        )

    return ExperimentSetup(MLSetup(X_train, y_train), MLSetup(X_test, y_test))


class CrossValidation:
    """A class to perform cross validated evaluation and training for any given ml model."""

    def __init__(
        self,
        score_function: Callable,
        test_fraction: float,
        n_splits: int = 5,
        random_seed: int = 10,
    ) -> None:

        # set a default split function, and stratify field.
        self._split_method = StratifiedShuffleSplit
        self._stratify_split = True

        self.scorer = score_function
        self.cv_splits = n_splits
        self.rnd = np.random.RandomState(random_seed)
        self.test_frac = test_fraction

    def set_split_function(self, split_function: Callable) -> None:
        logger.info(f"Set the cv split method as {split_function}")
        self._split_method = split_function

    def set_stratify(self, stratify: bool) -> None:
        logger.info(f"Set the split stratify flag as {stratify}")
        self._stratify_split = stratify

    def generate_splitter(self) -> Any:
        """
        Generate a sklearn cv split model for the dataset.

        Parameters
        ----------
        split_function: Any
            The function to use to generate the cv_splits. Expecting some sort of sklearn split generator, but can be any.

        Returns
        -------
            Any  (Returns the performed cv_split creator.)
        """
        return self._split_method(
            n_splits=self.cv_splits, test_size=self.test_frac, random_state=self.rnd
        )

    def train_model(
        self,
        model: Any,
        dataset: MLSetup,
        parameter_space: Dict[str, List[Union[int, float, str]]],
        random_search: bool = True,
        n_iterations: int = 30,
    ) -> Any:
        """
        Perform cross validated model training.

        Parameters
        ----------
        model: Any
            This is a model that you would like to train. Can be ANY model that has at minimum a fit, and a predict method.

        dataset: MLSetup
            This is the named tuple containing the train and test data.

        random_search: bool
            A boolean flag to delineate between random cv_search, and grid search. Default is random.

        n_iterations: int
            The number of random iterations to perform if random cv search is chosen. Default is 30

        Return
        ------
            Any  (The best trained model over all iterations)
        """

        # First, define the cv splits to use.
        cv_splitter = self.generate_splitter()
        split_indices = list(cv_splitter.split(dataset.obs, dataset.labels))

        # Now, setup each model iterations, either as random search or grid search. But default to grid search if less than n_iterations.
        setup_count = 1
        for parameter_values in parameter_space.values():
            setup_count *= len(parameter_values)

        if random_search and setup_count > n_iterations:
            setups = self.get_random_search_setups(parameter_space, n_iterations)
        else:
            if setup_count < n_iterations:
                logger.info(
                    f"Because there are less possible options ({setup_count}) than desired iterations ({n_iterations}), doing grid search."
                )
            setups = self.get_grid_search_setups(parameter_space)

        # Next, iterate over the setups and compute the score
        model_scores = [
            self.validated_train(model, dataset, split_indices, setup, i)
            for i, setup in enumerate(setups)
        ]

        # ... get the best scoring setup, and retrain over all data.
        best_score_idx = np.argmin(model_scores)
        best_model = model(**setups[best_score_idx])
        best_model.fit(dataset.obs, dataset.labels)
        return best_model

    def get_random_search_setups(
        self,
        parameter_space: Dict[str, List[Union[int, float, str]]],
        n_iterations: int,
    ) -> List[Dict[str, Union[float, int, str]]]:
        """
        Create the parameter definitions over a parameter space randomly

        Parameters
        ----------
        parameter_space: Dict[str, List[Union[int, float, str]]]
            A dictionary containing the parameter space setup, where the key is the parameter name, and the value is an Iterable of possible values.

        n_iterations: int
            The number of random setups to generate

        Return
        ------
        List[Dict[str, Union[float, int, str]]]
        """

        # Create the setups
        setups = []
        for _ in range(n_iterations):
            setup = {
                param: self.rnd.choice(vals) for param, vals in parameter_space.items()
            }
            setups.append(setup)

        return setups

    def get_grid_search_setups(
        self, parameter_space: Dict[str, List[Union[int, float, str]]]
    ) -> List[Dict[str, Union[float, int, str]]]:
        """
        Create the parameter definitions over in a grid.

        Parameters
        ----------
        parameter_space: Dict[str, List[Union[int, float, str]]]
            A dictionary containing the parameter space setup, where the key is the parameter name, and the value is an Iterable of possible values.


        Return
        ------
        List[Dict[str, Union[float, int, str]]]
        """

        # First create the combinations...
        options = [values for values in parameter_space.values()]
        possible_setups = product(*options)

        # ... and log how many setups there are...
        setups = 1
        for parameter_values in options:
            setups *= len(parameter_values)
        logger.info(
            f"Created {setups} setups to evaluate when griding the parameter space."
        )

        # ... then structure this for output as a list of dicts when returned
        parameter_names = list(parameter_space.keys())
        return [
            {parameter_names[i]: value for i, value in enumerate(setup)}
            for setup in possible_setups
        ]

    def validated_train(
        self,
        model: Any,
        data: MLSetup,
        splits: List[np.ndarray],
        params: Dict[str, Union[int, float, str]],
        iteration: int,
    ) -> float:
        """
        Perform training over random splits of the training data.

        Parameters
        ----------
        model : Any
            The model class you would like to train.
        data : MLSetup
            The data that you would like to use for the CV eval. (In this case the test data).
        splits : List[np.ndarray]
            A list of the indices that define the specific splitting of the data.
        params : Dict[str, Union[int, float, str]]
            A dictionary defining the specific parameters of the model to be trained with.
        iteration : int
            The current iteration of the CVSearch -- only used for logging.

        Returns
        -------
        float: The median score of the model trained and evaluated over all cv splits.
        """

        # Setup the model to train
        scores = []
        for split in splits:
            model_setup = model(**params)
            if not hasattr(model_setup, "fit"):
                raise AttributeError(
                    f"The model {model_setup} has no `.fit()` method to call. Make sure your model class has a `.fit` method."
                )

            # Get the split specific dataset...
            cv_train_obs, cv_train_labels = (
                data.obs.iloc[split[0]],
                data.labels.iloc[split[0]],
            )
            cv_test_obs, cv_test_labels = (
                data.obs.iloc[split[1]],
                data.labels.iloc[split[1]],
            )

            logger.info(
                f"\nCV Training over features ({cv_train_obs.columns}) and {len(cv_train_obs)} examples.\nTesting over {len(cv_test_obs)} examples.\n"
            )

            # ... then train the model and score.
            model_setup.fit(cv_train_obs, cv_train_labels)
            predictions = model_setup.predict(cv_test_obs)
            scores.append(self.scorer(cv_test_labels, predictions))

        result_score = np.median(scores)
        logger.info(
            f"Median score for iteration {iteration} {model_setup} is: {result_score}"
        )
        return result_score

    def validated_eval(
        self,
        predictions: np.ndarray,
        data: MLSetup,
        metric: Callable,
    ) -> CVEval:
        """
        Perform training over random splits of the training data.

        Parameters
        ----------
        predictions : np.ndarray
            The entire set of predictions made on the test set.
        data : MLSetup
            The data that you would like to use for the CV eval. (In this case the test data).
        metric : Callable
            The metric function to use for evaluation.

        Returns
        -------
        CVEval : A named tuple data structure of the result, "dot" indexed with mean, median, and std.
        """

        # First, define the cv splits to use.
        cv_splitter = self.generate_splitter()
        split_indices = list(cv_splitter.split(data.obs, data.labels))

        # Setup the model to train
        scores = []
        for split in split_indices:
            # Get the split specific dataset...
            cv_test_labels = data.labels.iloc[split[1]]

            scores.append(metric(cv_test_labels, predictions[split[1]]))

        result_median = np.median(scores)
        result_mean = np.mean(scores)
        result_std = np.std(scores)

        return CVEval(result_mean, result_median, result_std)
