# mlexpy
Simple utilities for handling and managing exploratory and experimental machine learning development.

## Introduction: 

### Design principles:
1. `mlexpy` is _not_ meant to be a tool deployed in a production prediction environment. Alternatively, it is meant to provide a simple structure to organize different components of machine learning to simplify basic ML exploration and expiramentation, and hopefully improve ML results via standardized and reproducable expiramentation. 

The core goal is to leverage fairly basic, yet powerful, and clear data strucutres and patterns to improve the "workspace" for ML development. Hopefully, this library can tirvialize some common ML development tasks, to allow developers, scientist, and any one else to spend more time in the _investigations_ in ML, and less time in coding or developing a reliable, readable, and reproduceable exploratory codebase / script / notebook.

`mlexpy` provides no explicit ML models or data. Instead it provides various tools to store, interact, and wrap differnet models, methods, or datasets.

#### High level library goals:
- 1. Provide intuitive, standardizeable, and reproduceable ML expiraments.
- 2. Methodological understandability is more important that method simplicity and efficiency. 
    - Becuase this is meant to aid in ML development, often times it is easy to lose track of what explcity steps are and were done in ultimately producing a predicion. `mlexpy` is meant to reduce the amount of code written for ML develpment purely to the explit steps taken in developing a model. For example, Pands DataFrames are currently preffered over numpy nd.arrays simply for column readability. Thus, verbosity is prefered. Ideally, by dumping a majority of the behind the scenes code to `mlexpy` this verbosity is still manageable.
    - Ideally, `mlexpy` makes is simple to understand exactly what a ML pipeling and model are doing, easing collaboration between engineers, coleuges, and academics.
- 3. `mlexpy` is not developed (yet?) for usage in large scale deep-learning tasks. Perhaps later this will be on interest.

2. `mlexpy` leverages an OOP framework, while this may be less intuitive for some practitioners, the benefits of becoming familiar with some OOP outweigh its learning curve.

3. The three junctions of the ML development pipeline are (1) the data -> (2) the processing -> (3) the predictions. Thus, each one of these steps is meant to be identically reproduceable independent of each other (to the extent possible). 
    - For example: I have an update to my dataset -- I would like to know **exactly** how a previously developed and stored pipeline and model will perform on this new dataset.
    - I have a new model, I would like to know how it performs given **exactly** the same data and processing steps.
    - I would like to try computing a feature differently in the processing steps. However, to compare to previous models, I would like to use **exactly** the same data, **exactly** the same process for the other features, and **exactly** the same model and training steps (ex. the same Cross Validation, parameter serching, and tuning).

    - Note: If the used features change, or the provided features change in all downstream process must change too in order to accomidate the structural change in the process, and no-longer can **exact** comparisons of a single component (data, process, model) be compared.


Note: Currently, `mlexpy` _only_ provides tooling for supervised learning.

## Structure
`mlexpy` is made up of 2 core modules, `processor` and `expirament`. These 2 modules interact to provide a clean space for ML development, with limited need for boilerplate "infrastructure" code.

#### Expected usage:
At minumum using `mlexpy` requries defining 2 classes each one with a `.process_data()` method. These classes inherit the `mlexpy` base classes, which don't have these methods defined, however, they do provide a variety of boilerplate ML tooling. An example workflow is shown below:

The plain language goals of `mlexpy` usage is described below. 
1. Do all of your data processing in the `processor` module. This is done by creating a child class that inherits the `PipelineProcessorBase` class, which bring in a wide variety of ML tooling. However, teh users need to write the `.process_data()` method in their child class.

2. Do all of your model expiramentation in the `{Classifier, Regresson}ExpiramentBase` class. However, a user needs to define the `process_data()` method for this class too. In general, this method should create an instance of the users pipeline child class and call the _pipeline_ class's `.process_data()` method.

These 2 methods provide the minumum needed infrastrucure from a user to perform model training and evaluation in a highly structured manner, as outlined in the principls and goals above.

- An example simple case of using `mlexpy` can be found in `examples/0_classification_example.ipynb`

##### Example pseudo-code
```
# Get mlexpy modules
from mlexpy.processor import ProcessPipelineBase
from mlexpy.experiment import ClassifierExperimentBase

# (1) Load data
dataset = <do your loading here>

# (2) Define your class to perform data processing
class SimpleProcessor(PipelineProcessorBase):
    def __init__(self, <all initilization arguments>)
        super().__init(<all initilization arguments>)

    def process_data(self):

        # Do a copy of the passed df
        df = df.copy()

        # Create some feature
        df["new_feature] = df["existing_feature"] * df["other_existing_feature"]

        # and return the resulting df
        return df

# (3) Define your class to do the expirament
class SimpleExperiment(ClassifierExperimentBase):
    def __init__(self, <all initilization arguments>)
        super().__init(<all initilization arguments>)
    
    def process_data(
        self, process_method_str: Optional[Callable] = "process_data"
    ) -> pipeline_utils.ExperimentSetup:

        processor = SimpleProcessor(process_tag=self.process_tag, model_dir=self.model_dir)

        # Now do the data processing on the method defined in process_method_str.
        process_method = getattr(processor, process_method_str)
        train_df = process_method(self.training.obs, training=True)
        test_df = process_method(self.testing.obs, training=False)

        print(
            f"The train data are of size {train_df.shape}, the test data are {test_df.shape}."
        )

        assert (
            len(set(train_df.index).intersection(set(test_df.index))) == 0
        ), "There are duplicated indecies in the train and test set."

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

(4) Now you can run your experiments using a random forest for example.

# Define the expirament
simple_experiment = SimpleExperiment(
    train_setup=dataset.train_data,
    test_setup=dataset.test_data,
    cv_split_count=20,
    model_tag="example_development_model",
    process_tag="example_development_process",
    model_dir=Path.cwd()
)

# Now begin the expiramentation, start with performing the data processing...
processed_datasets = simple_experiment.process_data()

# ... then train the model...
trained_model = simple_experiment.train_model(
    RandomForestClassifier(),
    processed_datasets,
    # model_algorithm.hyperparams,  # If this is passed, then cross validation search is performed, but slow.
)

# Get the predictions and evaluate the performance.
predictions = expirament.predict(processed_datasets, trained_model)
results =expirament.evaluate_predictions(processed_datasets, predictions=predictions)
```

### Data strcutures and functions
`mlexpy` uses a few namedtuple datastructures for storing data. These are used to act as immutable objects that contain training and test splits, that can be easily undersood via the named fields and dot notation.

- `MLSetup` 
Is a named tuple meant to store data for an ML prediction "run". It stores 2 variables:
    - `MLSetup.obs` storing the actual features dataframe
    - `MLSetup.labels` storing the respective ground truth values

- `ExperimentSetup` 
Is a higher level structure meant to provide all data for an expirament. Using the `mlexpy.pipeline_utils.get_stratified_train_test_data()` function will return an `ExperimentSetup` named tuple, containing 2 `MLExpirament` named tuples:
    - `ExperimentSetup.train_data` the ***training*** features and labels (stored as an `MLSetup`)
    - `ExperimentSetup.test_data` the ***testing*** features and labels (stored as an `MLSetup`)

- `pipeline_utils.get_stratified_train_test_data(train_data: pd.DataFrame, label_data: pd.Series, random_state: np.random.RandomState, test_frac: float = 0.3) -> ExperimentSetup:`
Performs a simple training at testing split of the data, however by default stratifies the dataset.

- `utils.initial_filtering(df: pd.DataFrame, column_mask_functions: Dict[str, List[Callable]]) -> pd.DataFrame:`
A simple method to perform any known filtering of the dataset prior to any training to testing split that might be desired, ex. droping nans, removing non-applicable cases, droping duplicates, etc.

To try to simplify this task, and reduce boiler plate needed code, this function only needs a dictionary with values that provide boolean outputs of `True` on the records to keep for a given column. Any record with a value of `False` will be dropped from the dataset. An example dictionary is shown below is shown below:

    ```
    drop_patterns = ["nan", "any-other-substring-i-dont-want"]
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
The `processor` module is meant to provide a clean space to perofrm any processing / feature engineering, is a structured manner. At minimum, this translates to defining the `.process_data()` method. A description of the critical methods required are provided below. A method that requires to be overriden by the child class wil raise a `NotImplementedError`.

#####  `.process_data(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:`
This method performs your feature engineering. A suggested template is:
    
    ```
    def process_data(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """Perform here all steps of the data processing for feature engineering."""
        logger.info(f"Begining to process all data of size {df.shape}.")

        # First, do anything needed to drop known "unwanted" examples...

        <...>

        # ... next, do an "hard-coded" transformations or feature engineering, meaning something that does not need any models (such as PCA, statistical scaling)...

        <...>
        logger.info(f"... processing complete for training={training}.")

        # ... now, handle any cases where features require a model (such as PCA, statistical scaling) to be trained, that are now appropraite to be trained on the testing data...
        if training:
            self.fit_model_based_features(feature_df)
            feature_df = self.transform_model_based_features(feature_df)
            self.dump_feature_based_models()  # this is optional
        else:
            # becuase we are training, we need to use the models already trained.
            feature_df = self.transform_model_based_features(feature_df)

        return feature_df
    ```

#####  `.fit_model_based_features(self, df: pd.DataFrame) -> None:`
This method performs all of your fitting of models to be used in model based features. Once fit, these models are stored in an `ordereddictionary` with dataframe column names as keys, and the list of the models to apply as a list as the values. This dictioary applies models in the exact same order as they were fit -- this way the steps of a pipeline can be preserved. An example is provided showing how a standard scaler is fit for every numerical column, and what the `.fit_scaler()` method might look like.

    ```
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

### `{Classification, Regression}ExpiramentBase` module
The `{Classification, Regression}ExpiramentBase` modlues are meant to provide a clean space to perofrm model training, and is a structured manner. the classification class and the regression classes are developed for each problem type. For the remainder of this description, a classification case will be applied, however is similar to regression. Similar to the `ProcessPipelineBase` class, again a `.process_data()` method needs to be defined. This method however, acts as a higher level wrapper for the Pipeline `.process_data()` methods governing both train set and test set processing, label encoding, and ensureing any observation - label index matching. A suggested template is:

#### `.process_data(self, process_method: Optional[str] = None) -> ExperimentSetup:`
This method performs all data processing for _both_ the training and testing data. The `process_method_str` argument is the name of the method you would like the processor class (in the example below `YourPipelineChildClass`) to use to process the data. By default this will be `.process_data()` however does not need to be. In this manner you can expirament with different pipeline processing methods, and store them in code, by simple passing different names inside of this function.
    ```
    def process_data(
        self, process_method_str: Optional[Callable] = "process_data"
    ) -> pipeline_utils.ExperimentSetup:

        processor = YourPipelineChildClass(process_tag=self.process_tag, model_dir=self.model_dir)

        # Now do the data processing on the method defined in process_method_str.
        process_method = getattr(processor, process_method_str)
        train_df = process_method(self.training.obs, training=True)
        test_df = process_method(self.testing.obs, training=False)

        print(
            f"The train data are of size {train_df.shape}, the test data are {test_df.shape}."
        )

        assert (
            len(set(train_df.index).intersection(set(test_df.index))) == 0
        ), "There are duplicated indecies in the train and test set."

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
    ```

### Notes:
More detailed documentation can be found in the examples, docs, and docstrings.

### Roadmap / TODOs:
- Expand to `numpy.ndarray`s?






