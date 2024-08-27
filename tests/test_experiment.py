import pytest
from mlexpy import experiment
from mlexpy.pipeline_utils import ExperimentSetup, MLSetup
import pandas as pd
from pathlib import Path
import sys


@pytest.fixture
def good_custom_model():
    # Simply a custom class model

    class CustomModel:
        def __init__(self) -> None:
            return

        def fit(self) -> float:
            return 0.0

        def predict(self) -> float:
            return 0.0

    return CustomModel()


@pytest.fixture
def bad_custom_model():
    # Simply a custom class model

    class CustomModel:
        def __init__(self) -> None:
            return

        def fit_model(self) -> float:
            return 0.0

        def make_prediction(self) -> float:
            return 0.0

    return CustomModel()


@pytest.fixture
def empty_data():
    return ExperimentSetup(MLSetup(None, None), MLSetup(None, None))


def test_basic_processor_exceptions(bad_custom_model, good_custom_model, empty_data):
    """Test that things don't work without defining the base processor as expected."""

    exp = experiment.ExperimentBase(train_setup=None, test_setup=None)
    bad_model = bad_custom_model
    good_model = good_custom_model

    # We can't call a processor...
    with pytest.raises(
        NameError,
        # match="The self.pipeline attribute has not be set. Run the .set_pipeline(<your-pipeline-class>) method to set the pipeline before processing.",
    ):
        # Assert that we raise a NameError here b/c we need to set the pipeline first.
        exp.process_data()

    # We can't call a stored model processor...
    with pytest.raises(
        NameError,
        # match="The self.pipeline attribute has not be set. Run the .set_pipeline(<your-pipeline-class>) method to set the pipeline before processing.",
    ):
        # Assert that we raise a NameError here b/c we need to set the pipeline first.
        exp.process_data_from_stored_models()

    with pytest.raises(AttributeError):
        # Assert that we can not call train on our bad model.
        exp.one_shot_train(
            bad_model, data_setup=empty_data, parameters={"placeholder": [1.0]}
        )

    with pytest.raises(AttributeError):
        # Assert that we can not call predict on our bad model.
        exp.predict(data_setup=empty_data, ml_model=bad_model)

    with pytest.raises(AttributeError):
        # Assert that we can not call predict_proba on our bad model.
        exp.predict(data_setup=empty_data, ml_model=bad_model, proba=True)

    with pytest.raises(ValueError):
        # Assert that the cv_train method must have an iterable parameter space.
        exp.cv_train(good_model, empty_data, parameters={"a": 0.5})

    with pytest.raises(ValueError):
        # Assert that the cv_train method must have an iterable parameter space that is NOT a string.
        exp.cv_train(good_model, empty_data, parameters={"a": "abcd"})


def test_directory_functions():
    """Test that we can successfully define our file structure"""

    exp = experiment.ExperimentBase(
        train_setup=None,
        test_setup=None,
        process_tag="test_example",
        model_dir=Path(__file__),
    )

    # Assert that we define the path as expected
    assert exp.model_dir == Path(__file__) / "test_example"

    exp_string = experiment.ExperimentBase(
        train_setup=None,
        test_setup=None,
        process_tag="test_example",
        model_dir=str(Path(__file__)),
    )

    # Assert that we create the correct path even if passing a string as the model directory
    assert exp_string.model_dir == Path(__file__) / "test_example"

    exp_none = experiment.ExperimentBase(
        train_setup=None,
        test_setup=None,
        process_tag="test_example",
    )

    # Assert that we create the correct path even if not passing a model directory
    assert exp_none.model_dir == Path(sys.path[-1]) / ".models" / "test_example"


def test_metric_add_remove():
    metric_name = "test_metric"

    def metric_fn(labels=[1, 2, 3], predictions=[0, 2, 3]):
        return sum([abs(label - predictions[i]) for i, label in enumerate(labels)])

    exp = experiment.ExperimentBase(
        train_setup=None,
        test_setup=None,
        process_tag="test_example",
        model_dir=Path(__file__),
    )

    # First, assert that there is no actual metrics in the base class
    assert len(exp.metric_dict) == 0

    exp.add_metric(metric_fn, metric_name)
    # Assert that this metric is now in the dictionary
    assert exp.metric_dict[metric_name] == metric_fn

    exp.remove_metric(metric_name)
    # Assert that the metric dict is empty again
    assert len(exp.metric_dict) == 0
