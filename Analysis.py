import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the file
df = pd.read_csv("high_diamond_ranked_10min.csv")

# Print the dataframe just to make sure its working properly
print(df)

# Print total number of elements in each column of the dataframe
# Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.count.html
print("\nThis is the total number of elements in each column of the dataframe:")
print(df.count())

# Checking the Dataset for any Nulls
# Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sum.html
print("\nThis is the number of nulls in the Dataset:")
print(df.isnull().sum())

# As you can see from the range of values, we may have to implement StandardScaler when using our models to reduce variance
print("\nChecking if Standard Scaler will be needed:")
print(df.head(6))

# Plotting Correlation Matrix using Pearson's Correlation Coefficent
# Ref: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
plt.figure(figsize=(24,12))
correlation=df.corr()
sns.heatmap(correlation,annot=True)
plt.title("Correlation Matrix",size=15)
plt.show()

# Firstly removing multicolinearity features then removing correlatory features with target feature
plt.figure(figsize=(22,12))
df = df.drop(columns=['redGoldDiff','redExperienceDiff','blueDeaths','redGoldDiff','blueKills','blueGoldDiff','blueExperienceDiff','redFirstBlood','redKills','blueAvgLevel','blueTotalGold','blueTotalExperience','redDeaths','redEliteMonsters','redAssists','redDragons','blueEliteMonsters','blueAssists','blueTotalMinionsKilled','blueCSPerMin','redTotalMinionsKilled','redCSPerMin','redAvgLevel','redGoldPerMin'])
correlation=df.corr()
sns.heatmap(correlation,annot=True)
plt.title("Final Correlation Matrix",size=20)
plt.show()

# Preparation of a new dataframe, converting floats to their string counterparts
old_values = []
string_values = []
for x in df['blueWins']:
    if x == 0:
        string_values.append("Blue Loss")
        old_values.append(0)
    elif x == 1:
        string_values.append("Blue Win")
        old_values.append(1)

# Forming a Dataframe for the representational bar chart
representational_dataframe = pd.DataFrame({'Old': old_values, 'New': string_values}).sort_values('New')

# Plotting Bar Chart to show distribution of results
# 4949 Blue Losses
# 4930 Blue Wins
sns.countplot(x=representational_dataframe.New, data=representational_dataframe)
plt.xlabel("End Game Result")
plt.ylabel("Number of Occurrences")
plt.show()