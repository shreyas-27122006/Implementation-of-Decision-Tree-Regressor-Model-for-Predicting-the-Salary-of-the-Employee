# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the employee dataset and separate input features and target variable (salary).
2. Split the dataset into training and testing sets.
3. Train the model using a Decision Tree Regressor with the training data.
4. Predict the employee salary using the test data and display the result.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SHREYAS M
RegisterNumber: 25013237
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv("C:/Users/acer/Downloads/Salary.csv")

X = data.iloc[:, 1:2].values   
y = data.iloc[:, 2].values    


regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

predicted_salary = regressor.predict([[6.5]])
print("Predicted Salary:", predicted_salary)


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Decision Tree Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

```

## Output:
<img width="1038" height="569" alt="image" src="https://github.com/user-attachments/assets/090eb399-6ad6-4dbe-a82a-3e42febccbe6" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
