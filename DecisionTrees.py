import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
#-----------------------------------------------------------------------------------------------------#
# Line Graph Function to generate graphs for certain hyperparamters.
def line_graph_for_hyperparameter(x,y):
    accuracy_values = []
    hyperparameter_value = []
    for i in range(x,y):
        # Initialize the Decision Tree (default no max depth)
        decision_tree_model = DecisionTreeClassifier(max_depth=i)
        decision_tree_model.fit(X_train, y_train)
        predictions = decision_tree_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        #Append to array
        accuracy_values.append(accuracy*100)
        hyperparameter_value.append(i)

    #Initializing Plot
    graph_plot = pd.DataFrame({'Number': hyperparameter_value, 'Accuracy': accuracy_values})
    sns.set(style="darkgrid")
    sns.lineplot(x=graph_plot.Number, y=graph_plot.Accuracy)
    plt.title("Max Depth vs Accuracy (Decision Trees)")
    plt.xlabel("Maximum Depth Value of Decision Tree")
    plt.ylabel("Accuracy (as a percentage)")
    plt.show()

    # Dropping low values and amending the graph_plot dataframe
    graph_plot = graph_plot[graph_plot.Accuracy==max(graph_plot.Accuracy)]
    print("Your best accuracy in the line graph was :", graph_plot['Accuracy'].values[0], "this was when i was", graph_plot['Number'].values[0])
#-----------------------------------------------------------------------------------------------------#
# Baseline Model
def baseline_model():

    # Initialize the Decision Tree (default no max depth)
    decision_tree_model = DecisionTreeClassifier()

    # Fit/Train the Decision Tree
    decision_tree_model.fit(X_train, y_train)

    # Making a prediction
    predictions = decision_tree_model.predict(X_test)

    # Printing the accuracy
    accuracy = accuracy_score(y_test,predictions)
    print("Your baseline accuracy rating for Decision Trees is:", accuracy*100 ,"%")

    # Calculating how much the model has missed the targets
    misclassification_rate = 1 - accuracy
    print("Your error rate is", misclassification_rate * 100, "%")

    # Printing a Confusion Matrix and changing the x ticks and y ticks to String Variables
    x_axis_labels = ["Blue Loss","Blue Win"]
    confmatrix = confusion_matrix(y_test, predictions)
    sns.heatmap(confmatrix, annot=True, cmap="Greens", fmt='d', xticklabels=x_axis_labels, yticklabels=x_axis_labels)
    plt.title("Initial Confusion Matrix Decision Trees")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
#-----------------------------------------------------------------------------------------------------#
# Best Model
def best_model():

    # Initialize the Decision Tree
    decision_tree_model = DecisionTreeClassifier(min_samples_split=0.1)

    # GridSearch for max_depth (with a specific range in mind), max_features and min_samples_leaf
    grid = {"max_depth":[3,4,5,6],"max_features": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],"min_samples_leaf": [0.1,0.2,0.3,0.4,0.5,1]}
    final_decision_tree_model = GridSearchCV(decision_tree_model, grid)

    # final_decision_tree_model.fit(X_train, y_train)
    final_decision_tree_model.fit(X_train, y_train)

    # Making a prediction
    predictions = final_decision_tree_model.best_estimator_.predict(X_test)

    # Printing the accuracy
    accuracy = final_decision_tree_model.best_score_
    print("Your final accuracy rating is for Decision Trees is:", accuracy * 100, "%")

    # Calculating how much the model has missed the targets
    misclassification_rate = 1 - accuracy
    print("Your error rate is", misclassification_rate * 100, "%")

    print("These are the parameters for the best model:", final_decision_tree_model.best_params_)

    # Printing a Confusion Matrix and changing the x ticks and y ticks to String Variables
    x_axis_labels = ["Blue Loss", "Blue Win"]
    confmatrix = confusion_matrix(y_test, predictions)
    sns.heatmap(confmatrix, annot=True, cmap="Greens", fmt='d', xticklabels=x_axis_labels, yticklabels=x_axis_labels)
    plt.title("Final Confusion Matrix Decision Trees")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Tree doesn't take up full space REMEMBER TO ADD TITLE IN WORD
    plt.figure(figsize=(25, 25))
    plot_tree(final_decision_tree_model.best_estimator_)
    plt.tight_layout()
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

# REMEMBER TO COMMENT THIS OUT TO DO SEPARATE TESTS
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Baseline Model. remember to comment out standard scaler
baseline_model()

# Running the best model
best_model()

# Line Graph Generator. Currently Set to Max Depth
line_graph_for_hyperparameter(1,30)

# PLEASE NOTE THESE SCORES WERE ACCURATE AT THE TIME OF TESTING BUT PLEASE REFER TO THE MAIN REPORT IF YOU WANT OUR FINAL ACCURACY SCORES
# Baseline accuracy without Standard Scaler Accuracy is: 62.79%
# Baseline accuracy with Standard Scaler: 62.6% - DO NOT use Standard Scaler
# Adjusting hyper parameter and using criterion "entropy" reduces accuracy from 63.29% (gini) to 62.07% (entropy)
# Noticed using splitter "random" hyperparameter, decreased accuracy to: 61.84% - USE DEFAULT SPLITTER "best"
# Setting "class_weight to balanced made no difference to accuracy.
# Decided to use GridSearchCV in order to determine the number of max_features and min_samples_leaf as it is too unpredictable
# Min_samples_split 0.1 best accuracy after that gets lower. 71.89%
# Max_Depth tends to be of around 4 with highest accuracy of 70.78% but can sometimes change so use Grid Search on this as well

# FINAL Model should consist of NO standard scaler, gini, best splitter, grid search with max_features and min_samples_leaf and max_depth, no CV as that causes accuracy to go down. 0.1 for min_min_samples_split
