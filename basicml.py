from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
diabetes=datasets.load_diabetes()


import matplotlib.pyplot as plt
# import  matpltlib.pyplot as pltI

diabetes_x=diabetes.data
print(diabetes_x.shape)
diabetes_x_train=diabetes_x[0:-30]
diabetes_x_test=diabetes_x[-30:]
# print(diabetes_x_test.shape)
daibetes_y_train=diabetes.target[0:-30]
daibetes_y_test=diabetes.target[-30:]

model=linear_model.LinearRegression()
model.fit(diabetes_x_train,daibetes_y_train)
a=model.predict(diabetes_x_test)
print("mean sqr error ",mean_squared_error(daibetes_y_test,a))
# print(diabetes_x_train)
print("weights: ",model.coef_)
print("intercepts: ",model.intercept_)
