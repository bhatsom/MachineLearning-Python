import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#  A telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. 
#  If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. 
#  It is a classification problem. That is, given the dataset, with predefined labels, we need to build a model to be used to predict class of a new or unknown case.

# The example focuses on using demographic data, such as region, age, and marital, to predict usage patterns.
# The target field, called custcat, has four possible service categories that correspond to the four customer groups, as follows:
# 1) Basic Service 2) E-Service 3) Plus Service 4) Total Service

# Our objective is to build a classifier to predict the service category for unknown cases.

## Load Data
# read the data using pandas library and print the first five rows.
df = pd.read_csv('teleCust1000t.csv')
print(df.head())

## Data Visualization and Analysis
# Let's first look at the class-wise distribution of the data set.
# We can see that 281 customers opt for Plus Services, 266 for Basic-services, 236 for Total Services, and 217 for E-Services. 
# Thus the data set seems mostly balanced between the different classes and requires no special means of accounting for class bias.
print(df['custcat'].value_counts())

# We can also visualize the correlation map of the data set to determine how the different features are related to each other.
correlation_matrix = df.corr()
print(correlation_matrix)
plt.figure(figsize=(10, 8))
print(sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5))

# From the correlation map, some features have better correlation among them than others, indicating the depth of relationship between the two features.
# Important thing is the correlation of the target feature, i.e. custcat with all the other features. This will help us identify which features 
# should be focussed on for modeling and which ones can be ignored. Lets get a list of features sorted in the descending order of their absolute correlation 
# values with respect to the target field.

correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
print(correlation_values)
# Above shows that the features retire and gender have the least effect on custcat while ed and tenure have the most effect.

## Separate Input and Target Features
X = df.drop('custcat',axis=1)
y = df['custcat']

## Normalize Data
# Data normalization is important for KNN model as it makes predictions based on the distance between data points (samples), i.e. for a given test point, 
# the algorithm finds the k-nearest neighbors by measuring the distance between the test point and other data points in the dataset. By normalizing / standardizing 
# the data, we ensure that all features contribute equally to the distance calculation. 

# Since normalization scales each feature to have zero mean and unit variance, it puts all features on the same scale (with no feature dominating due to its larger range).
# This helps KNN make better decisions based on the actual relationships between features, not just on the magnitude of their values.

X_norm = StandardScaler().fit_transform(X)

## Train Test Split
# Retain 20% of the data for testing and the rest for training. Assigning a random state ensures reproducibility of the results across multiple executions.
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

## KNN Classification
# Once the data is in place, we can now execute the training of the model.

# Training - initially, we may start with a small value as the value of k, say k = 3
k = 3
# Train Model and Predict  
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)

# Predicting - Once the model is trained, we can now use this model to generate predictions for the test set
yhat = knn_model.predict(X_test)

# Accuracy evaluation 
# In multilabel classification, accuracy classification score is a function that computes subset accuracy. This function is equal to the jaccard_score function. 
# Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.
print("Test set Accuracy: ", accuracy_score(y_test, yhat))
