import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


df = pd.read_csv("FuelConsumption.csv") 
#how it sounds

#take a look at the dataset
df.head()

#summarize the data
# describe() is used to view some basic statistical details like percentile, mean, std etc.
df.describe()
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#You can select a column (df[col]) and return column with label col as Series or a few columns (df[[col1, col2]])
cdf.head(9)

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
#The hist() function in pyplot module of matplotlib library is used to plot a histogram.

plt.show()
# scatter them in a plot
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')

plt.xlabel("FUELCONSUMPTION_COMB")
#xlabel() function pyplot module of matplotlib libary is used to set the label for the x-axis 
plt.ylabel("Emission")
#ylabel() This function sets the label for the y-axis of the plot.
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')

plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

msk = np.random.rand(len(df)) < 0.8
#function creates an array of specified shape and fills it with random values.
train = cdf[msk]
test = cdf[~msk]
#For example if a=60 (0011 1100 in binary) its complement is -61 (-0011 1101) stored in 2's complement



plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
#numpy.asanyarray()function is used when we want to convert input to an array
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
#fitting is equal to training. Then, after it is trained, the model can be used to make predictions, usually with a .predict() method call.
#The fit-method is always to learn something in machine learning.
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
#i think this is the meaning.
#In this case, the coefficient is the slope of the fitted line, and the intercept is the point where the fitted line intersects with the y-axis.
#For retrieving the slope (coefficient of x): print(regressor.coef_)
#To retrieve the intercept: print(regressor.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

#intercept_: float or array of shape (n_targets,)

#coef_ : array of shape (n_features, ) or (n_targets, n_features)


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )

#- Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
#Mean Squared Error (MSE)
#Root Mean Squared Error (RMSE).