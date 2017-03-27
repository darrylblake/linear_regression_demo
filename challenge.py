import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import random

dataframe = pd.read_csv('challenge_dataset.txt', names=['X', 'Y'])
x_values = dataframe[['X']]
y_values = dataframe[['Y']]

x_reg = linear_model.LinearRegression()
x_reg.fit(x_values, y_values)

count = x_values.count()['X']
prediction_differences = []
for index in range(count):
    prediction = x_reg.predict(x_values.values[index][0])[0][0]
    actual = y_values.values[index][0]
    prediction_differences.append(abs(prediction - actual))

dataframe['Prediction Difference'] = pd.Series(prediction_differences)    
print(dataframe)

plt.scatter(x_values, y_values)
plt.scatter(prediction_differences, y_values)
plt.plot(x_values, x_reg.predict(x_values))
plt.show()
