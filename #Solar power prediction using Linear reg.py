## Solar power prediction using Linear regression model
#importing necessary libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#loading the dataset
df = pd.read_csv('solarpowergeneration.csv')

#display the first few rows of dataset
df.head()
#checking the total number of rows and columns
df.shape
df.tail() #showing last five rows of the dataset
#display summary
df.describe()
#check the info of dataset
df.info()
#checking the missing values
df.isnull().sum()
#checking the duplicated values
df.duplicated().sum()
#plot distribution of power
plt.figure(figsize = (10, 6))
sns.histplot(df['generated_power_kw'], bins=30, kde=True)
plt.title("Distribution of Generated Power (kw)")
plt.xlabel('Generated Power (kw)')
plt.ylabel("Frequency")
plt.show()
# EDA
df[df.columns[0:9]].hist(bins=30, figsize=(15, 10))
plt.show()
df[df.columns[9:18]].hist(bins=30, figsize=(15, 10))
plt.show()
df[df.columns[18:20]].hist(bins=30, figsize=(6, 3))
plt.show()
# Bivariate analysis
# Scatter plot with target feature
plt.figure(figsize=(15, 30))
for i, column in enumerate(df.columns):
    plt.subplot(7, 3, i+1)
    plt.scatter(df[column], df['generated_power_kw'])
    plt.title(f'{column} vs Generated power(kW)')
    plt.xlabel(column)
    plt.ylabel('Generated power(kW)')
plt.tight_layout()
plt.show()
df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), cmap = 'coolwarm', annot= True, fmt = '.2f')
# Outlier
plt.figure(figsize = (15, 30))
for i,column in enumerate(df.columns):
    plt.subplot(7, 3, i+1)
    sns.boxplot(df[column])
plt.show()

# Building the machine learning model to predict solar power output generation
#pip install scikit-learn
# Linear regression model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
df.columns
# Splitting the dataset into features and targets
X = df.drop('generated_power_kw', axis = 1)
y = df['generated_power_kw']

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
# Evaluate the model - test
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error on Test Set: {mae}")
# Evaluate the model on the training set
y_pred_train = model.predict(X_train_scaled)
mae_train = mean_absolute_error(y_train, y_pred_train)
print(f"Mean Absolute Error on Train Set: {mae_train}")
