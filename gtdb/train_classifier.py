import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
import sys
import pickle

def train_model(gt_file):

    data = np.genfromtxt(gt_file, delimiter=',')

    height, width = data.shape

    X = data[:, :width-1]
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

    # Choose the type of classifier.
    clf = RandomForestClassifier(class_weight='balanced',
                                 criterion='entropy',
                                 max_depth=10,
                                 max_features='auto',
                                 min_samples_leaf=1,
                                 min_samples_split=2,
                                 n_estimators=50)

    # Choose some parameter combinations to try
    # parameters = {'n_estimators': [4, 6, 9, 50],
    #               'max_features': ['log2', 'sqrt','auto'],
    #               'criterion': ['entropy', 'gini'],
    #               'max_depth': [2, 3, 5, 10],
    #               'min_samples_split': [2, 3, 5],
    #               'min_samples_leaf': [1, 5, 8]
    #              }


    #{'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
    # 0.971423403870501

    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    # Run the grid search
    #grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
    #grid_obj = grid_obj.fit(X_train, y_train)

    #print(grid_obj.best_params_)

    # Set the clf to the best combination of parameters
    #clf = grid_obj.best_estimator_

    # Fit the best algorithm to the data.
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print('test accuracy', accuracy_score(y_test, predictions))

    clf.fit(X, y)
    predictions = clf.predict(X)
    print('all data train accuracy', accuracy_score(y, predictions))

    with open('random_forest.pkl', 'wb') as pickle_file:
        pickle.dump(clf, pickle_file)

if __name__ == "__main__":

    gt_file = sys.argv[1]
    train_model(gt_file)
