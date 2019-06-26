import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
import sys
import pickle

def train_model(gt_file):

    data = np.genfromtxt(gt_file, delimiter=',')

    X = data[:, :10]
    y = data[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

    # Choose the type of classifier.
    clf = RandomForestClassifier()

    # Choose some parameter combinations to try
    # parameters = {'n_estimators': [4, 6, 9, 50],
    #               'max_features': ['log2', 'sqrt','auto'],
    #               'criterion': ['entropy', 'gini'],
    #               'max_depth': [2, 3, 5, 10],
    #               'min_samples_split': [2, 3, 5],
    #               'min_samples_leaf': [1, 5, 8]
    #              }

    parameters = {'n_estimators': [50],
                  'max_features': ['log2', 'sqrt', 'auto'],
                  'criterion': ['entropy', 'gini'],
                  'max_depth': [2, 3, 5, 10],
                  'min_samples_split': [2, 3, 5],
                  'min_samples_leaf': [1, 5, 8],
                  'class_weight': ['balanced']
                  }

    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    # Run the grid search
    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, y_train)

    # Set the clf to the best combination of parameters
    clf = grid_obj.best_estimator_

    # Fit the best algorithm to the data.
    clf.fit(X_train, y_train)

    with open('random_forest.pkl', 'wb') as pickle_file:
        pickle.dump(clf, pickle_file)

    predictions = clf.predict(X_test)
    print(accuracy_score(y_test, predictions))

if __name__ == "__main__":

    gt_file = sys.argv[1]
    train_model(gt_file)
