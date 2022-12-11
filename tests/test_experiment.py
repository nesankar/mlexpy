import pytest
from mlexpy import experiment
import pandas as pd
from pathlib import Path
import sys


def test_basic_processor_exceptions():
    """Test that things don't work without defining the base processor as expected."""

    exp = experiment.ExperimentBase(train_setup=None, test_setup=None)

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
