import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
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
    - according to docs, for classification problems max_features="sqrt" is a good default
    - the out of bag error comes as a free unbiased estimation of the performance of the model
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