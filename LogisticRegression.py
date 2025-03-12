import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# -----------------------------------------------------------------------------------------------------#
# Baseline Model
def baseline_model():
    # Splitting the data using train_test_split
    # when doing document, put in random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Standard Scaler is being used on X_train and then the transform is being used on X_test to reduce variance and make processing times quicker
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Preparation of a Logistic Regression Model. Default Max Iter is 100
    logistic_regression_model = LogisticRegression()

    # Fitting the data
    logistic_regression_model.fit(X_train, y_train)

    # Making a prediction
    predictions = logistic_regression_model.predict(X_test)

    # Printing the accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("Your baseline accuracy rating is for Logistic Regression is:", accuracy * 100, "%")

    # Calculating how much the model has missed the targets
    misclassification_rate = 1 - accuracy
    print("Your error rate is", misclassification_rate * 100, "%")

    # Printing a Confusion Matrix and changing the x ticks and y ticks to String Variables
    x_axis_labels = ["Blue Loss", "Blue Win"]
    confmatrix = confusion_matrix(y_test, predictions)
    sns.heatmap(confmatrix, annot=True, cmap="Blues", fmt='d', xticklabels=x_axis_labels, yticklabels=x_axis_labels)
    plt.title("Confusion Matrix Logistic Regression with Standard Scaler")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# -----------------------------------------------------------------------------------------------------#
# Function for K-fold cross validation
def k_fold_version(number_of_folds):
    # Preparation of a Logistic Regression Model. Default Max Iter is 100
    logistic_regression_model = LogisticRegression()

    # Splitting using K-Fold Cross Validation
    accuracy = 0
    loops = 0

    # Preparation of standard scaler
    scaler = StandardScaler()

    # 5 Folds
    kfolds = KFold(number_of_folds)
    for train_index, test_index in kfolds.split(X):
        # Standard Scaler is applied
        X_train, X_test = scaler.fit_transform(X.iloc[train_index]), scaler.transform(X.iloc[test_index])
        y_train, y_test = y[train_index], y[test_index]
        logistic_regression_model.fit(X_train, y_train)
        predictions = logistic_regression_model.predict(X_test)
        accuracy += (accuracy_score(y_test, predictions))
        loops = loops + 1

    # Calculating the accuracy
    accuracy = accuracy / loops

    # Printing the accuracy
    print("Your accuracy rating is for Logistic Regression with", number_of_folds, "Fold Cross Validation is:",
          accuracy * 100, "%")

    # Calculating how much the model has missed the targets
    misclassification_rate = 1 - accuracy
    print("Your error rate with ", number_of_folds, "Fold Cross Validation is:", misclassification_rate * 100, "%")


# -----------------------------------------------------------------------------------------------------#
def best_test_model():
    # Splitting the data using train_test_split
    # when doing document, put in random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Standard Scaler is being used on X_train and then the transform is being used on X_test to reduce variance and make processing times quicker
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Preparation of a Logistic Regression Model. Default Max Iter is 100
    logistic_regression_model = LogisticRegression()

    grid = {"solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}
    final_logistic_regression_model = GridSearchCV(logistic_regression_model, grid, cv=5)

    final_logistic_regression_model.fit(X_train, y_train)

    # Making a prediction
    predictions = final_logistic_regression_model.best_estimator_.predict(X_test)

    print("best solver is:", final_logistic_regression_model.best_params_)

    # Printing the accuracy
    accuracy = final_logistic_regression_model.best_score_
    print("Your final accuracy rating is for Logistic Regression is:", accuracy * 100, "%")

    # Calculating how much the model has missed the targets
    misclassification_rate = 1 - accuracy
    print("Your error rate is", misclassification_rate * 100, "%")


# -----------------------------------------------------------------------------------------------------#
# Adding data
df = pd.read_csv("high_diamond_ranked_10min.csv")

# Dropping the correlated features
df = df.drop(columns=['redGoldDiff', 'redExperienceDiff', 'blueDeaths', 'redGoldDiff', 'blueKills', 'blueGoldDiff',
                      'blueExperienceDiff', 'redFirstBlood', 'redKills', 'blueAvgLevel', 'blueTotalGold',
                      'blueTotalExperience', 'redDeaths', 'redEliteMonsters', 'redAssists', 'redDragons',
                      'blueEliteMonsters', 'blueAssists', 'blueTotalMinionsKilled', 'blueCSPerMin',
                      'redTotalMinionsKilled', 'redCSPerMin', 'redAvgLevel', 'redGoldPerMin'])

# Storing the features in X
X = df.drop(columns='blueWins')

# Setting the target variable
y = df['blueWins']

# Calling baseline model
baseline_model()
# Calling k fold model with 5 fold
k_fold_version(5)
# Calling best test with 5 fold and finding the correct solver
best_test_model()
# -----------------------------------------------------------------------------------------------------#
# PLEASE NOTE THESE SCORES WERE ACCURATE AT THE TIME OF TESTING BUT PLEASE REFER TO THE MAIN REPORT IF YOU WANT OUR FINAL ACCURACY SCORES
# Baseline accuracy without any iterations and WITHOUT STANDARD SCALER: 48.04%
# Accuracy without any iterations and WITH STANDARD SCALER: 72.06%
# Accuracy of 5 Fold Cross Validation resulted in: 72.84%
# Adjusting solver hyper parameters using GridSearchCV with 5 Fold CV:
# Changing the max_iter parameter going up in increments of 10 up to 100 did not make a difference
# Changing the solver increased accuracy to: newton-cg resulted in the BEST accuracy sometimes of >73 (73.3% was max) but generally around mid 72%