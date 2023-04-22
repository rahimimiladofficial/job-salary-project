# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 10:43:49 2023

@author: Milad Rahimi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# import data
df = pd.read_csv('eda_data.csv')

# choose relevant columns
df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','Company_txt','hourly','employer_provided',
             'job_state','same_state','age','python_yn','spark_yn','aws_yn','excel_yn','title_simp','seniority','desc_length']]

# get dummy data
df_dum = pd.get_dummies(df_model)

# train test split 
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis =1)
y = df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# statsmdel as a linear regression 
import statsmodels.api as sm

x_sm = X = sm.add_constant(X)
model = sm.OLS(y, x_sm)
model.fit()
#print(model.fit().summary())

# sickit-learn as a linear regression 
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

reg = LinearRegression().fit(X_train, y_train)
np.mean(cross_val_score(reg, X_train, y_train, scoring = 'neg_mean_absolute_error', cv= 3))

# lasso regression 
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

# tuning the model
alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha, error)


err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]

# random forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))

# tune models GridsearchCV 
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,100,10), 'criterion':('friedman_mse','absolute_error'), 'max_features':('sqrt', 'log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_
