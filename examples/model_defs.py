from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor


from collections import namedtuple

MlModelInfo = namedtuple(
    "MlModelInfo",
    ["model", "hyperparams"],
)


def randomforest_info(classifier: bool = True) -> MlModelInfo:
    """Return the info needed for a random forest model training run."""

    if classifier:
        model = RandomForestClassifier
    else:
        model = RandomForestRegressor

    params = {
        "n_estimators": (5, 10, 25, 50, 100, 150, 250, 500, 750),
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [2, 5, 10, 20, 25, 50, 100, 200],
        "min_samples_split": [2, 5, 10, 20, 25, 40],
        "max_features": ["sqrt", "log2", None, 1, 2, 5, 10, 20],
    }

    return MlModelInfo(model, params)


def sgdmodel_info(classifier: bool = True) -> MlModelInfo:
    """Return the info needed for a stochastic gradient descent model  training run."""

    if classifier:
        model = SGDClassifier
    else:
        model = SGDRegressor

    params = {
        "loss": [
            "log_loss",
            "modified_huber",
            "squared_hinge",
            "perceptron",
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ],
        "penalty": ["l1", "l2", "elasticnet"],
        "alpha": [
            0.000001,
            0.00001,
            0.0001,
            0.001,
            0.01,
            0.1,
            0.2,
            0.5,
            0.7,
            0.9,
            1,
            2,
            3,
            5,
        ],
        "l1_ratio": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        "fit_intercept": [True, False],
        "shuffle": [True, False],
        "epsilon": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
    }

    return MlModelInfo(model, params)


MODEL_DICT = {
    "randomforest": randomforest_info,
    "sgd": sgdmodel_info,
}
