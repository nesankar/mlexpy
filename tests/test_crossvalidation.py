import pytest
from fixtures import simple_binary_dataframe, simple_series, simple_dataframe
from mlexpy.pipeline_utils import CrossValidation, MLSetup
from pandas.testing import assert_frame_equal
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error

f1_metric = lambda labels, preds: -f1_score(labels, preds, average="macro")

rmse = lambda labels, preds: np.sqrt(mean_squared_error(labels, preds))


@pytest.fixture
def model_definitions():

    return {
        "param1": [1, 2, 3, 4, 5],
        "param2": [1.2, 1.3, 1.4, 0.1],
        "param3": ["a", "b", "c"],
    }


def get_simple_cv(cv_split_frac, random_iterations, random_seed, score_function):

    random_state = np.random.RandomState(random_seed)

    return CrossValidation(
        test_fraction=cv_split_frac,
        score_function=score_function,
        n_splits=random_iterations,
        random_seed=random_state.get_state(legacy=False)["state"]["key"][
            -1
        ],  # needs to be an integer here
    )


def test_cv_splitting(simple_binary_dataframe):
    """Test that basic functionality of the cross validation tooling."""

    # First, test that the same seed results in the same splits.
    cv1 = get_simple_cv(0.5, 5, 10, f1_metric)
    splitter_1 = cv1.generate_splitter()

    cv2 = get_simple_cv(0.5, 5, 10, f1_metric)
    splitter_2 = cv2.generate_splitter()

    cv3 = get_simple_cv(0.5, 5, 100, f1_metric)
    splitter_3 = cv3.generate_splitter()

    # assert splitter_1 == splitter_2
    # "The splitter objects are NOT the same when initializing the SAME cv_splitter object."
    # assert splitter_1 != splitter_3
    # "The splitter objects ARE the same when initializing DIFFERENT cv_splitter objects."

    # Next, test that the actual data are different or the same if desired.
    dataobj = MLSetup(
        simple_binary_dataframe[["obs1", "obs2"]], simple_binary_dataframe["target"]
    )

    idcs_1 = list(splitter_1.split(dataobj.obs, dataobj.labels))
    idcs_2 = list(splitter_2.split(dataobj.obs, dataobj.labels))
    idcs_3 = list(splitter_3.split(dataobj.obs, dataobj.labels))

    print(idcs_1[0][0])
    print(idcs_2[0][0])
    print(idcs_3[0][0])

    assert all([i_1 == idcs_2[0][0][i] for i, i_1 in enumerate(idcs_1[0][0])])
    "The splitter data are NOT the same when initializing the SAME cv_splitter object."
    assert any([i_1 != idcs_3[0][0][i] for i, i_1 in enumerate(idcs_1[0][0])])
    "The splitter data ARE the same when initializing DIFFERENT cv_splitter objects."

    # Test that we correctly set the startify flag
    cv3.set_stratify(True)
    assert cv3._stratify_split == True


def test_parameter_space(model_definitions):

    cv1 = get_simple_cv(0.5, 5, 10, f1_metric)
    cv2 = get_simple_cv(0.5, 5, 10, f1_metric)
    cv3 = get_simple_cv(0.5, 5, 100, f1_metric)

    space_1 = cv1.get_random_search_setups(model_definitions, 5)
    space_2 = cv2.get_random_search_setups(model_definitions, 5)
    space_3 = cv3.get_random_search_setups(model_definitions, 5)

    # Test that we don't create the same parameter spaces when we do and dont want to.
    assert space_1 == space_2
    "The parameter space options are NOT the same when initializing the SAME cv_splitter object."
    assert space_1 != space_3
    "The parameter space options ARE the same when initializing DIFFERENT cv_splitter objects."

    # Test that we create the correct number of options using grid search.
    option_count = 1
    for values in model_definitions.values():
        option_count *= len(values)

    grid_space = cv1.get_grid_search_setups(model_definitions)

    assert option_count == len(grid_space)
    "The number of setups in the grid search result is not equal to all possible setups."
