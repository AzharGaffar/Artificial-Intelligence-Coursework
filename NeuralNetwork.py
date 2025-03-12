import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
#-----------------------------------------------------------------------------------------------------#
# Baseline Model
def baseline_model():
    sq_model = Sequential()
    sq_model.add(Dense(200, activation='relu')) # Hidden layer with activation function Relu
    sq_model.add(Dense(100, activation='relu')) # Hidden layer with activation function Relu
    sq_model.add(Dense(50, activation='relu')) # Hidden layer with activation function Relu
    sq_model.add(Dense(25, activation='relu')) # Hidden layer with activation function Relu
    sq_model.add(Dense(12, activation='relu')) # Hidden layer with activation function Relu
    sq_model.add(Dense(5, activation='relu')) # Hidden layer with activation function Relu
    sq_model.add(Dense(1, activation='sigmoid')) # Hidden layer with activation function Sigmoid
    # Compile with Adam Optimizer
    sq_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics='accuracy')

    # Fitting the baseline model with just 400 epochs
    sq_model.fit(x=X_train, y=y_train, epochs=400)

    # Saving the Predictions in a prediction variable. (The terminal said the old way was deprecated)
    predictions = (sq_model.predict(X_test) > 0.5).astype("int32")

    # Printing the Accuracy Score of the baseline
    accuracy = accuracy_score(y_test, predictions)*100
    print("The Accuracy of your baseline multilayer perceptron model is:", accuracy ,"%")
#-----------------------------------------------------------------------------------------------------#
# This is the intermediate of our reapproach
def intermediate_model():
    sq_model = Sequential()
    sq_model.add(Dense(16, activation='relu')) # Hidden layer with activation function Relu
    sq_model.add(Dense(8, activation='relu')) # Hidden layer with activation function Relu
    sq_model.add(Dense(4, activation='relu')) # Hidden layer with activation function Relu
    sq_model.add(Dense(2, activation='relu')) # Hidden layer with activation function Relu
    sq_model.add(Dense(1, activation='sigmoid')) # Hidden layer with activation function Sigmoid
    # Compile with Adam Optimizer
    sq_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics='accuracy')

    # Declaring an early stop when accuracy is at its max
    es = EarlyStopping(monitor='accuracy', mode='max', patience=25)

    # Fitting the intermediate level model with an early stop
    sq_model.fit(x=X_train, y=y_train, epochs=400, callbacks=[es])

    # Saving the Predictions in a prediction variable. (The terminal said the old way was deprecated)
    predictions = (sq_model.predict(X_test) > 0.5).astype("int32")

    # Printing the Intermediate Accuracy Score
    accuracy = accuracy_score(y_test, predictions)*100
    print("The Accuracy of your intermediate multilayer perceptron model is:", accuracy, "%")
#-----------------------------------------------------------------------------------------------------#
# This is the final model for our multilayer perceptron. Implementation of dropouts and changing the optimizer
def final_model():
    sq_model = Sequential()
    sq_model.add(Dense(16, activation='relu')) # Hidden layer with activation function Relu
    # sq_model.add(Dropout(0.2)) Commented out because its not needed anymore
    sq_model.add(Dense(8, activation='relu')) # Hidden layer with activation function Relu
    sq_model.add(Dropout(0.4)) # Dropout layer
    sq_model.add(Dense(4, activation='relu')) # Hidden Layer with activation function Relu
    sq_model.add(Dropout(0.4)) # Dropout Layer
    # sq_model.add(Dense(2, activation='relu')) Commented out because its not needed anymore
    sq_model.add(Dense(1, activation='sigmoid')) # Final Layer with activation function sigmoid
    #Optimizer changed to SGD
    sq_model.compile(loss='binary_crossentropy', optimizer='SGD', metrics='accuracy')

    # Declaring an early stop that monitors when accuracy is at its max
    es = EarlyStopping(monitor='accuracy', mode='max', patience=25)

    # Fit the model with 400 epochs and use the early stop
    sq_model.fit(x=X_train, y=y_train, epochs=400, callbacks=[es])

    # PLot the loss and the accuracy to show how the model generalizes well
    plot_loss_and_accuracy = pd.DataFrame(sq_model.history.history)
    plot_loss_and_accuracy.plot()
    plt.title("Accuracy and Loss vs Epoch Number")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss and Accuracy")
    plt.show()

    # Saving the Predictions in a prediction variable. (The terminal said the old way was deprecated)
    predictions = (sq_model.predict(X_test) > 0.5).astype("int32")

    # Printing the Final Accuracy Score
    accuracy = accuracy_score(y_test, predictions)*100
    print("The Accuracy of your final multilayer perceptron model is:", accuracy, "%")

    # The final confusion matrix
    x_axis_labels = ["Blue Loss", "Blue Win"]
    confmatrix = confusion_matrix(y_test, predictions)
    sns.heatmap(confmatrix, annot=True, cmap="Greys", fmt='d', xticklabels=x_axis_labels, yticklabels=x_axis_labels)
    plt.title("Final Confusion Matrix Multilayer Perceptron")
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

# Splitting the data using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Standard Scaler is being used to reduce variance and make processing times quicker
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Please uncomment if you want to run this
# baseline_model()

# Please uncomment if you want to run this
# intermediate_model()

#This is the final model
final_model()

# Baseline overfitting and not very efficient
# Intermedate uses a different approach in terms of the number of layers and features
# Final uses the best approach alongside dropouts and changing the optimizer







