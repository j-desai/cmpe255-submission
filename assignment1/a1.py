#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest as skb, chi2, f_regression
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
boston = load_boston()     
b = pd.read_csv('housing.csv')
column_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
b = pd.read_csv('housing.csv', names = column_name, sep = "\s+")
X_full = b.iloc[:,:13]
y_full = b.iloc[:,13:]
# Selectin feature LSTAT
feature = ['LSTAT']
b_feature = b[feature].copy()
b_feature['MEDV'] = y_full['MEDV']
b_feature.head()
b_feature = b_feature.sort_values(by='LSTAT')
X_lin = b_feature.iloc[:,:1]
y_lin = b_feature.iloc[:,1:]
def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)

X_train, X_test, y_train, y_test = train_test_split(X_lin, y_lin, test_size = 0.3, random_state = 32)
lin = LinearRegression()
lin.fit(X_train,y_train)
y_pred= lin.predict(X_test)
print("R2 Score = " , r2_score(y_test, y_pred))
print("\n RMSE = ", rmse(y_test,y_pred))
plt.scatter(X_lin,y_lin)
plt.plot(X_test,y_pred, color= 'red')
plt.show()

#Polynomial Regression for degree = 2

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_lin)
poly_reg.fit(X_poly,y_full)
lin.fit(X_poly,y_lin)
ypoly_pred=lin.predict(poly_reg.fit_transform(X_lin))
rmse(y_lin,ypoly_pred)
print("\n RMSE for Polynomial Regression degree 2 = " , rmse(y_lin, ypoly_pred))
print("\n R2 Score for Polynomial Regression degree 2 = " , r2_score(y_lin, ypoly_pred))
plt.scatter(X_lin,y_lin)
plt.plot(X_lin,ypoly_pred, color = 'red')
plt.show()

#Polynomial Regression for degree = 20

poly_reg = PolynomialFeatures(degree=20)
X_poly = poly_reg.fit_transform(X_lin)
poly_reg.fit(X_poly,y_full)
lin.fit(X_poly,y_lin)
y_pred=lin.predict(poly_reg.fit_transform(X_lin))
rmse(y_lin,y_pred)
print("\n RMSE for Polynomial Regression degree 20= " , rmse(y_lin, y_pred))
print("\n R2 Score for Polynomial Regression degree 20 = " , r2_score(y_lin, y_pred))
plt.scatter(X_lin,y_lin)
plt.plot(X_lin,y_pred, color = 'red')
plt.show()

# ## Multiple Regression

features_multi = ['INDUS','RM','PTRATIO','LSTAT']
b_features_multi = b[features_multi].copy()
b_features_multi['MEDV'] = y_full['MEDV']
X_mul = b_features_multi.iloc[:,:4]
y_mul = b_features_multi.iloc[:,4:]
Xmul_train, Xmul_test, ymul_train, ymul_test = train_test_split(X_mul, y_mul, test_size = 0.3, random_state = 32)

mul_lin = LinearRegression()
mul_lin.fit(Xmul_train,ymul_train)
ymul_pred = mul_lin.predict(Xmul_test)
print("\n R2 Score for Multiple Regression = " , r2_score(ymul_test, ymul_pred))
print("\n RMSE for Multiple Regression = ",rmse(ymul_test,ymul_pred))
adjR2 = 1 - (1-mul_lin.score(Xmul_test, ymul_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("\n Adjusted R-squared score for Multiple Regression = ", adjR2)



