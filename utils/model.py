"""
Scope:
    Model definition
    Best parameter estimation
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def compute_rf_selection(X, y, n_estimators_set=[], max_features_set=[], n_folds=5):
    """
    Arguments:
        X: data
        y: labels
        n_estimators_list: list of n_estimators to choose from
        max_features_list: list of max_features to choose from
        n_folds: number of "corss validations" using oob accuracy
    Returns:
        best_model, model_performances dict
    """
    model_performances = []
    best_accuracy, best_model = 0, None
    for n_estimators in n_estimators_set:
        for max_features in max_features_set:
            accuracy = 0
            for _ in range(n_folds):
                clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, oob_score=True, bootstrap=True)
                clf.fit(X, y)
                accuracy += clf.oob_score_
            accuracy /= n_folds
            model_performances.append({"accuracy_oob": accuracy, "n_estimators": n_estimators, "max_features": max_features})
    performances = [performance["accuracy_oob"] for performance in model_performances]
    i_performances = np.argsort(performances)[:: -1]
    best_model = model_performances[i_performances[0]]
    return best_model, model_performances

def compute_svm_selection(X, y, parameters):
    """
    Arguments:
        X: data
        y: labels
        parameters: 
    Returns:
        best model where parameters are chosen using grid search
    """
    svc = SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X, y)
    return clf.best_params_