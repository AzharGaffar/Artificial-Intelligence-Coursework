import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
#-----------------------------------------------------------------------------------------------------#
# Function to check for optimum K values
def plot_k_values(starting, ending):
    accuracy_list = []
    n_neighbors_list = []
    for x in range(starting, ending):
        # In the default K Nearest Neighbors Algo, the default is 5
        knn_model = KNeighborsClassifier(n_neighbors=x)
        knn_model.fit(X_train, y_train)
        predictions = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) * 100
        accuracy_list.append(accuracy)
        n_neighbors_list.append(x)

    # Making a dataframe to display the plot and also easier to get the max from this
    n_neighbors_df = pd.DataFrame({'NeighborsValue': n_neighbors_list, 'Accuracy': accuracy_list})

    # Plotting everything
    sns.set(style="darkgrid")
    sns.lineplot(x=n_neighbors_df.NeighborsValue, y=n_neighbors_df.Accuracy)
    plt.title("K value vs Accuracy (KNN)")
    plt.xlabel("n_neighbors Value (K Value)")
    plt.ylabel("Accuracy (as a percentage)")
    plt.show()

    # Finding out the max accuracy and the corresponding n_neighbor value
    n_neighbors_df = n_neighbors_df[n_neighbors_df.Accuracy == max(n_neighbors_df.Accuracy)]
    print("Your best accuracy was was:", n_neighbors_df['Accuracy'].values[0],"this was when n_neighbors hyperparameter was", n_neighbors_df['NeighborsValue'].values[0])
#-----------------------------------------------------------------------------------------------------#
def baseline_model():
    # Default K value is 5
    knn_model = KNeighborsClassifier()

    # Fitting the model
    knn_model.fit(X_train, y_train)

    # Making a prediction
    predictions = knn_model.predict(X_test)

    # Generating the accuracy of the model
    accuracy = accuracy_score(y_test, predictions)
    print("Your baseline accuracy rating for K Nearest Neighbors is:", accuracy * 100, "%")

    # Calculating how much the model has missed the targets
    misclassification_rate = 1 - accuracy
    print("Your error rate is", misclassification_rate * 100, "%")

    # Printing a Confusion Matrix and changing the x ticks and y ticks to String Variables
    x_axis_labels = ["Blue Loss", "Blue Win"]
    confmatrix = confusion_matrix(y_test, predictions)
    sns.heatmap(confmatrix, annot=True, cmap="Reds", fmt='d', xticklabels=x_axis_labels, yticklabels=x_axis_labels)
    plt.title("Initial Confusion Matrix K Nearest Neighbors")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
#-----------------------------------------------------------------------------------------------------#
def best_knn_model():
    # Initialize the knn model
    knn_model = KNeighborsClassifier(weights="distance",metric="manhattan")

    # Generating the grid k values values to test between
    grid_shortcut =[]
    for x in range(30,100):
        grid_shortcut.append(x)

    # This is the grid search part
    grid = {"n_neighbors": grid_shortcut}
    final_knn_model = GridSearchCV(knn_model, grid)

    # Fitting the model
    final_knn_model.fit(X_train, y_train)

    # Making a prediction with the best estimator
    predictions = final_knn_model.best_estimator_.predict(X_test)

    # Printing the accuracy
    accuracy = final_knn_model.best_score_
    print("Your final accuracy rating is for K Nearest Neighbors is:", accuracy * 100, "%")

    # Calculating how much the model has missed the targets
    misclassification_rate = 1 - accuracy
    print("Your error rate is", misclassification_rate * 100, "%")

    # Printing the final parameters for the best model possible
    print("These are the parameters for the best model:", final_knn_model.best_params_)

    # Printing a Confusion Matrix and changing the x ticks and y ticks to String Variables
    x_axis_labels = ["Blue Loss", "Blue Win"]
    confmatrix = confusion_matrix(y_test, predictions)
    sns.heatmap(confmatrix, annot=True, cmap="Reds", fmt='d', xticklabels=x_axis_labels, yticklabels=x_axis_labels)
    plt.title("Final Confusion Matrix K Nearest Neighbors")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
#-----------------------------------------------------------------------------------------------------#
# Adding data
df = pd.read_csv("high_diamond_ranked_10min.csv")

# Dropping the correlated features
df = df.drop(columns=['redGoldDiff','redExperienceDiff','blueDeaths','redGoldDiff','blueKills','blueGoldDiff','blueExperienceDiff','redFirstBlood','redKills','blueAvgLevel','blueTotalGold','blueTotalExperience','redDeaths','redEliteMonsters','redAssists','redDragons','blueEliteMonsters','blueAssists','blueTotalMinionsKilled','blueCSPerMin','redTotalMinionsKilled','redCSPerMin','redAvgLevel','redGoldPerMin'])

# Storing the features in X
X = df.drop(columns = 'blueWins')

# Setting the target variable
y = df['blueWins']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

# Standard Scaler is being used so that variance will be reduced and the data points can be classified easily
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Baseline model that we started with and started experimenting off. Please Uncomment this if you want to run it
# baseline_model()

# This is the best model that uses grid search with other hyperparameters
best_knn_model()

# Uncomment this to see the plot from 1 to 200 of K_values. Warning takes a long time
# Plotting the K values for the report and to analyse them
# plot_k_values(1,200)

# PLEASE NOTE THESE SCORES WERE ACCURATE AT THE TIME OF TESTING BUT PLEASE REFER TO THE MAIN REPORT IF YOU WANT OUR FINAL ACCURACY SCORES
# Baseline model, No Standard Scaling: Accuracy 50.708502024291505 %
# Baseline model, w/ Standard Scaling: Accuracy 65.65452091767881 %
# Improving Baseline Model:
    # First Hyperparameter to test: n_neighbors: goes up to low 70's in high k values
        # Obvious to go over 30 for this but do not go over 100 as trade off
        # Computationally expensive
        # Use gridsearch
    # In terms of weighting: distance hyperparameter makes a difference and causes accuracy to go to 65.72%
    # Algorithm: leave on auto didn't see difference when changing values
    # Metric: manhattan distance causes accuracy to go up to 67.13900134952766 %

# best model: 72.06073752711497 % CV no difference (used random state 33)
