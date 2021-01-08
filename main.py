import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import db, model

arguments = sys.argv
has_arguments = len(sys.argv) > 1

# Data loading
X_test, y_test = db.load_data("parkinsons_test.dt")
X_train, y_train = db.load_data("parkinsons_train.dt")

if not has_arguments or "1" in arguments:
    """
    Random forest predictions
        n_estimators: 750
        max_features: 4
    """
    # Find optimal parameters
    n_features = X_train.shape[1]
    n_estimators = [100, 250, 500, 750, 1000]
    max_features = [int(np.sqrt(n_features)), n_features]
    best_model, model_performances = model.compute_rf_selection(X_train, y_train, n_estimators_set=n_estimators, max_features_set=max_features)
    # Compute predictions
    clf = RandomForestClassifier(n_estimators=best_model["n_estimators"], max_features=best_model["max_features"], max_depth=None, bootstrap=True, oob_score=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("--Best model: {}".format(best_model))
    print("--Test accuracy: {}".format(accuracy))

if not has_arguments or "2" in arguments:
    """
    Support vector machines
        kernel: radial basis
    """
    log_scale = np.power(10.0, np.arange(-3, 4))
    parameters = {"kernel": ["rbf"], "C": log_scale, "gamma": log_scale}
    best_model = model.compute_svm_selection(X_train, y_train, parameters)
    clf = SVC(C=best_model["C"], gamma=best_model["gamma"], kernel="rbf")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("--Best model: {}".format(best_model))
    print("--Test accuracy: {}".format(accuracy))