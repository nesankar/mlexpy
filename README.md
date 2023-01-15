![master](https://github.com/nesankar/mlexpy/actions/workflows/master.yml/badge.svg)
[![codecov](https://codecov.io/gh/nesankar/mlexpy/branch/master/graph/badge.svg?token=2E7B8O1Q8F)](https://codecov.io/gh/nesankar/mlexpy)
[![CodeFactor](https://www.codefactor.io/repository/github/nesankar/mlexpy/badge/master)](https://www.codefactor.io/repository/github/nesankar/mlexpy/overview/master)
![pypi](https://img.shields.io/pypi/v/mlexpy.svg)
![versions](https://img.shields.io/pypi/pyversions/mlexpy.svg)
[![PyPI status](https://img.shields.io/pypi/status/mlexpy.svg)](https://pypi.org/project/mlexpy/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/mlexpy.svg)](https://badge.fury.io/py/mlexpy)



# mlexpy
Simple utilities for handling and managing exploratory and experimental machine learning development.

## Installation

`mlexpy` can be installed via:

```pip install mlexpy```

- As an alpha release, any input is highly encouraged and likely to be implemented. Please feel free to open an issue.

## Introduction: 

### Design principles:
1. `mlexpy` is _not_ meant to be a tool deployed in a production prediction environment. Alternatively, it is meant to provide a simple structure to organize different components of machine learning to simplify basic ML exploration and experimentation, and hopefully improve ML results via standardized and reproducible experimentation. 

The core goal is to leverage fairly basic, yet powerful, and clear data structures and patterns to improve the "workspace" for ML development. Hopefully, this library can trivialize some common ML development tasks, to allow developers, scientist, and any one else to spend more time in the _investigations_ in ML, and less time in coding or developing a reliable, readable, and reproducible exploratory codebase / script / notebook.

`mlexpy` provides no explicit ML models or data. Instead it provides various tools to store, interact, and wrap different models, methods, or datasets.

2. `mlexpy` is not meant to replace any work of a statistician or data scientist. But it is meant to improve their workflow with boilerplate code and "infrastructure" setup to perform better experiments faster.

#### High level library goals:
- 1. Provide intuitive, standardizable, and reproducible ML experiments.
- 2. Methodological understandability is more important that method simplicity and efficiency. 
    - Because this is meant to aid in ML development, often times it is easy to lose track of what explicitly steps are and were done in ultimately producing a prediction. `mlexpy` is meant to reduce the amount of code written for ML development purely to the explicit steps taken in developing a model. For example, Pandas DataFrames are currently preferred over numpy nd.arrays simply for column readability. Thus, verbosity is preferred. Ideally, by dumping a majority of the behind the scenes code to `mlexpy` this verbosity is still manageable.
    - Ideally, `mlexpy` makes is simple to understand exactly what a ML pipeling and model are doing, easing collaboration between engineers, coworker, and academics.
- 3. `mlexpy` is not developed (yet?) for usage in large scale deep-learning tasks. Perhaps later this will be on interest.

2. `mlexpy` leverages an OOP framework, while this may be less intuitive for some practitioners, the benefits of becoming familiar with some OOP outweigh its learning curve.

3. The three junctions of the ML development pipeline are (1) the data -> (2) the processing -> (3) the predictions. Thus, each one of these steps is meant to be identically reproducible independent of each other (to the extent possible). 
    - For example: I have an update to my dataset -- I would like to know **exactly** how a previously developed and stored pipeline and model will perform on this new dataset.
    - I have a new model, I would like to know how it performs given **exactly** the same data and processing steps.
    - I would like to try computing a feature differently in the processing steps. However, to compare to previous models, I would like to use **exactly** the same data, **exactly** the same process for the other features, and **exactly** the same model and training steps (ex. the same Cross Validation, parameter searching, and tuning).

    - Note: If the used features change, or the provided features change; all downstream process **must** change too in order to accommodate the structural change in the process, and no-longer can **exact** comparisons of a single component (data, process, model) be compared.


Note: Currently, `mlexpy` _only_ provides tooling for supervised learning.

## Structure
`mlexpy` is made up of 2 core modules, `processor` and `experiment`. These 2 modules interact to provide a clean space for ML development, with limited need for boilerplate "infrastructure" code.

#### Expected usage:
At minimum using `mlexpy` requires defining a class that defines a `.process_data()` method that inherits the `processor.ProcessPipelineBase` class. The `ProcessPipelineBase` class provides a variety of boilerplate ML tooling. An example workflow is shown below.

The plain language usage of `mlexpy`: 
1. Do all of your processing of raw data in `.process_data()` method of your `ProcessPipelineBase`'s child class.

2. Do all of your model experimentation via the `experiment` module. This is done  the `{Classifier, Regression}Experiment` class. The `{Classifier, Regression}Experiment` class manages your test and train sets, file naming conventions, and random seeds internally. However, the user needs to define the pipeline class to use via the `{Classifier, Regression}Experiment.set_pipeline()` method. 

These two classes provide the minimum needed infrastructure from a user to perform model training and evaluation in a highly structured manner, as outlined in the principles and goals above.

- An example simple case of using `mlexpy` can be found in `examples/0_classification_example.ipynb`

##### Example pseudo-code
```python
# Get mlexpy modules
from mlexpy.processor import ProcessPipelineBase
from mlexpy.experiment import ClassifierExperiment

# (1) Load data
dataset = <...do-your-loading-here...>

# (2) Define your class to perform data processing
class SimplePipeline(PipelineProcessorBase):
    def __init__(self, <all initialization arguments>)
        super().__init(<all initialization arguments>)

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:

        # Do a copy of the passed df
        df = df.copy()

        # Create some feature
        df["new_feature"] = df["existing_feature"] * df["other_existing_feature"]

        # and return the resulting df
        return df
# (3) Now you can run your experiments using a random forest for example.

# Define the experiment
simple_experiment = ClassifierExperiment(
    train_setup=dataset.train_data,
    test_setup=dataset.test_data,
    cv_split_count=20,
    model_tag="example_development_model",
    process_tag="example_development_process",
    model_dir=Path.cwd()
)

# Set the pipeline to use...
simple_experiment.set_pipeline(SimplePipeline)
# Now begin the experimentation, start with performing the data processing...
processed_datasets = simple_experiment.process_data()

# ... then train the model...
trained_model = simple_experiment.train_model(
    RandomForestClassifier(),
    processed_datasets,
    # params=model_algorithm.hyperparams,  # If this is passed, then cross validation search is performed, but slow.
)

# Get the predictions and evaluate the performance.
predictions = experiment.predict(processed_datasets, trained_model)
results = experiment.evaluate_predictions(processed_datasets, predictions=predictions)
```

### Data structures and functions
`mlexpy` uses a few namedtuple data structures for storing data. These are used to act as immutable objects that contain training and test splits, that can be easily understood via the named fields and dot notation.

- `MLSetup` 
Is a named tuple meant to store data for an ML prediction "run". It stores 2 variables:
    - `MLSetup.obs` storing the actual features dataframe
    - `MLSetup.labels` storing the respective ground truth values

- `ExperimentSetup` 
Is a higher level structure meant to provide all data for an experiment. Using the `mlexpy.pipeline_utils.get_stratified_train_test_data()` function will return an `ExperimentSetup` named tuple, containing 2 `MLExperiment` named tuples:
    - `ExperimentSetup.train_data` the ***training*** features and labels (stored as an `MLSetup`)
    - `ExperimentSetup.test_data` the ***testing*** features and labels (stored as an `MLSetup`)

- `pipeline_utils.get_stratified_train_test_data(train_data: pd.DataFrame, label_data: pd.Series, random_state: np.random.RandomState, test_frac: float = 0.3) -> ExperimentSetup:`
Performs a simple training at testing split of the data, however by default stratifies the dataset.

- `utils.initial_filtering(df: pd.DataFrame, column_mask_functions: Dict[str, List[Callable]]) -> pd.DataFrame:`
A simple function to perform any known filtering of the dataset prior to any training to testing split that might be desired, ex. dropping `nan`s, removing non-applicable cases, dropping duplicates, etc.

To simplify this task, and reduce boiler plate needed code, this function only needs a dictionary with values that provide boolean outputs of `True` on the records to keep for a given column. Any record with ***at least 1 value of `False` will be dropped from the dataset***. An example dictionary is shown below is shown below:

```python
DROP_PATTERNS = ["nan", "any-other-substring-i-dont-want"]
def response_to_drop(s: str) -> bool:
    """For any string, return True if any of the substrings in drop patterns are present in the string s."""
    return not any([drop_pattern in s for drop_pattern in DROP_PATTERNS])

def non_numeric_response(s: str) -> bool:
    """For any string, return True if the string is NOT of a numeric value. (ex not '2')"""
    return s.isnumeric()

column_filter_dict: Dict[str, List[Callable]] = {
    "response": [response_to_drop],
    "time_in_seconds": [non_numeric_response],
}
```

### `processor` module
The `processor` module is meant to provide a clean space to perform any processing / feature engineering, is a structured manner. At minimum, this translates to defining the `.process_data()` method. A description of the critical methods required are provided below. A method that requires to be overridden by the child class wil raise a `NotImplementedError`.

#####  `.process_data(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:`
This method performs your feature engineering. Note that this method (and supplementary provided methods) are made to easily handle processing when training, and when testing. A suggested template is:
    
```python
def process_data(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """Perform here all steps of the data processing for feature engineering."""

    # First, do an "hard-coded" transformations or feature engineering, meaning something that does not need any models (such as multiplying values, or domain specific calculations)...
    <...your-code-here...>

    # ... now, handle any cases where features require a model (such as PCA, statistical scaling, onehot-encoding) to be trained, that are not appropriate to be trained on the testing data...
    if training:
        self.fit_model_based_features(feature_df)
        feature_df = self.transform_model_based_features(feature_df)
        self.dump_feature_based_models()  # this is optional
    else:
        # because we are training, we need to use the models already trained.
        feature_df = self.transform_model_based_features(feature_df)

    # Return the 
    return feature_df
```

#####  `.fit_model_based_features(self, df: pd.DataFrame) -> None:`
This method performs all of your fitting of models to be used in model based features. Once fit, these models are stored in an `DefaultOrderedDictionary` with dataframe column names as keys, and the list of the models to apply as a list as the values. This dictionary applies models in the exact same order as they were fit -- this way the steps of a pipeline can be preserved. An example is provided showing how a standard scaler is fit for every numerical column, and what the `.fit_scaler()` method might look like.

```python
def fit_model_based_features(self, df: pd.DataFrame) -> None:
    """Here to model fitting for a transformation."""

    for column in df.columns:
        if df[column].dtype not in ("float", "int"):
            continue
        self.fit_scaler(df[column])

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
```

### `{Classification, Regression}Experiment` module
The `{Classification, Regression}Experiment` modules provide a clean space to perform model training in a structured manner. The classification class and the regression classes are developed for each problem type. For the remainder of this description, a classification case will be applied, however is similar to regression. The Experiment classes require a pipeline attribute to be set via the `.set_pipeline()` method, where you should pass your developed child of `ProcessPipelineBase` as an argument. Via the _Experiment_ class you can run `.process_data()` which will perform all processing defined in your `ProcessPipelineBase`'s child class's `.process_data()` method.

#### `.process_data(self, process_method: Optional[str] = None) -> ExperimentSetup:`
This method performs all data processing for _both_ the training and testing data. The `process_method_str` argument is the name of the method you would like the processor class (in the example below `YourPipelineChildClass`) to use to process the data. By default this will be `.process_data()` however does not need to be. In this manner you can experiment with different processing methods in your pipeline class, and store them in code, by simply passing different names inside of this function.


#### Model training in the `ExperimentBase` class:

Training an ML model is very simple using `mlexpy`. All of the necessary code for model training, and for _basic_ hyperparameter search is defined in the `ExperimentBase` class. A model can be trained via the `.train_model()` method. If a hyperparameter space is provided to this method, then training includes a cross validated search through the hyperparemeter space. In the example above, the following snipit performs model training:
```python
# ... then train the model...
trained_model = simple_experiment.train_model(
    RandomForestClassifier(),
    processed_datasets,
    # params=model_algorithm.hyperparams,  # If this is passed, then cross validation search is performed, but slow.
)
```

##### Hyperparameter tuning
By default, **if** a hyperparameter space is provided, A randomized search of 20 iterations with 5-fold cross validation will be performed. If using a classifier, the scoring function for cross validation will be `f1_macro`, and for regression will be `rmse` (taken as a negative for maximization).

### Evaluation
Each of the `{Classification, Regression}Experiment` classes define a relevant method for model evaluation, with the respective metrics. Each class type stores a dictionary of a metric_name -> metric_callable mapping, and evaluation simply operates for every key -> value pair in this dictionary. Custom metrics can be provided via the `.add_metric()` method, however need to accept the labels, and predictions as the input ex: `my_metric(labels: Iterable, predictions: Iterable)` **Note**: If predictions are passed as class probabilities, make sure your metric functions raises an error if only the predicted class is provided


### Notes:
More detailed usage can be found in the examples and docstrings including:

- `examples/0_classification_example.ipynb` A notebook showing a number of applications of `mlexpy`.
- `examples/1_scripted_example.py` Showing a rough outline for suggested file and modules structure using `mlexpy`.
- `examples/2_store_all_models_example.py` Showing how to call the model i/o tooling to store trained models, and to load trained models generating identical predictions.
- `examples/3_loading_models_example.py` Showing how to load stored model definitions to evaluate an existing method on "new" data.
- `examples/4_initial_filtering_and_binary_class_example.py` Showing how to use the initial filtering tooling (`mlexpy.utils.initial_filtering()`) and a case of binary classification.
- `examples/5_cv_search_for_hyperpareters.py` Showing how to perform hyperparameter search, and an example of a model_definition file to store ML model definitions.
- `examples/6_regression_pca_example.py`Showing how to perform a regression example, and how to add in a possible PCA method for dimensionality reduction.


### Roadmap / TODOs:
- Capabilities:
    - Test and/or expand to general statistical prediction models. (beyond the `sklearn` framework)
    - Add `xarray` integration
    - Add clustering 
    - Expand to `numpy.ndarray`s?
- Convenience:
    - Add in readthedocs







