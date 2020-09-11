import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading data
salaries = pd.read_csv('salaries.csv')

x = salaries.iloc[:,1:2]
y = salaries.iloc[:,2:]
X = x.values
Y = y.values


# Linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()


# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

# Prediction for degree = 4 
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

# Support Vector Regression
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_scalable = sc1.fit_transform(X)

sc2 = StandardScaler()
y_scalable = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

from sklearn.svm import SVR

svr_reg = SVR(kernel ='rbf')
svr_reg.fit(x_scalable, y_scalable)

plt.scatter(x_scalable, y_scalable, color= 'red')
plt.plot(x_scalable, svr_reg.predict(x_scalable), color='blue')


