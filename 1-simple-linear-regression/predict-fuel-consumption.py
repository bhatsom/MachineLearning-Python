import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline

# load the data
df = pd.read_csv("simple-linear-regression/FuelConsumptionCo2.csv")

# df operations wrapped in print() to display on terminal when not using Jupitar NoteBook

# View the first 5 rows
# df.head()
# print(df.head())

# verify successful load with some randomly selected records
#print(df.sample(5))

#print("\n")

# statistical summary of the data
print(df.describe())

# Select a few features that might be indicative of CO2 emission to explore more
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print(cdf) - prints all 1067 rows
print(cdf.sample(9))

# Visualize features - consider the histograms for each of these features.
# As you can see, most engines have 4, 6, or 8 cylinders, and engine sizes between 2 and 4 liters.
# As you might expect, combined fuel consumption and CO2 emission have very similar distributions.
viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()

# Go ahead and display some scatter plots of these features against the CO2 emissions, to see how linear their relationships are.
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
# This is an informative result. Three car groups each have a strong linear relationship between their combined fuel consumption and their CO2 emissions. 
# Their intercepts are similar, while they noticeably differ in their slopes.

# Lets try scatter plot for another item (ENGINESIZE) against CO2 emissions 
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.xlim(0,27)
plt.show()
# Although the relationship between engine size and CO2 emission is quite linear, their correlation is weaker than that for each of the three fuel 
# consumption groups. Notice that the x-axis range has been expanded to make the two plots more comparable.

# Extract the input feature and labels from the dataset
# Lets use engine size to predict CO2 emission with a linear regression model. Begin the process by extracting the input feature and 
# target output variables, X and y, from the dataset.

X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

# Create train and test datasets - randomly split data into train and test sets, using 80% for training and reserving the remaining 20% for testing.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# The outputs are one-dimensional NumPy arrays or vectors
print(type(X_train))
print(np.shape(X_train))

# Build a simple linear regression model
from sklearn import linear_model
# create a model object
regressor = linear_model.LinearRegression()
# train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1, 1), y_train)
# Here, Coefficient and Intercept are the regression parameters determined by the model. They define the slope and intercept of the 'best-fit' line to the training data.
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)

# Visualize model outputs
# Visualize the goodness-of-fit of the model to the training data by plotting the fitted line over the data.
# The regression model is the line given by: y = intercept + coefficient * x
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Model evaluation
# Compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics play a key role in the development of a model, 
# as they provide insight into areas that require improvement. There are different model evaluation metrics, let's use MSE here to calculate the accuracy of our model based on the test set:

# Mean Absolute Error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just an average error.

# Mean Squared Error (MSE): MSE is the mean of the squared error. In fact, it's the metric used by the model to find the best fit line, and for that reason, 
# it is also called the residual sum of squares.

# Root Mean Squared Error (RMSE). RMSE simply transforms the MSE into the same units as the variables being compared, which can make it easier to interpret.

# R-squared is not an error but rather a popular metric used to estimate the performance of your regression model. It represents how close the data points are to the fitted regression line. The higher the R-squared value, the better the model fits your data. The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
# Use the predict method to make test predictions
y_test_ = regressor.predict( X_test.reshape(-1,1))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test_, y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test_, y_test))
print("Root mean squared error: %.2f" % root_mean_squared_error(y_test_, y_test))
print("R2-score: %.2f" % r2_score( y_test_, y_test) )

