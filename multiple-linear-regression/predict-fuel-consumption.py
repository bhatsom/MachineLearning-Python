import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline

# load the data
df = pd.read_csv("../simple-linear-regression/FuelConsumptionCo2.csv")

# verify successful load with some randomly selected records
#print(df.sample(5))

# statistical summary of the data
print(df.describe())

# Drop categoricals and any unseless columns
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)

# Now that you have eliminated some features, take a look at the relationships among the remaining features.Analyzing a correlation matrix that displays 
# the pairwise correlations between all features indicates the level of independence between them. It also indicates how predictive each feature is of the target.
# You want to eliminate any strong dependencies or correlations between features by selecting the best one from each correlated group.

print(df.corr())

df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)
print(df.head(9))