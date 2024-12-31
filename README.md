# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5.Predict the values of arrays.
6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7.Predict the values of array.
8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: 
RegisterNumber:  
*/
```
```.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20,8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show() 
```
## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
![Screenshot 2024-11-28 103640](https://github.com/user-attachments/assets/4bb22582-591f-458f-9756-346c0bcd745e)
![Screenshot 2024-11-28 103648](https://github.com/user-attachments/assets/a5a7fafd-43fd-48a8-ad97-3a2c3f86598c)
![Screenshot 2024-11-28 103653](https://github.com/user-attachments/assets/9aa53115-fc64-4068-9e06-1bb1945f4ed7)
![Screenshot 2024-11-28 103704](https://github.com/user-attachments/assets/0d48dc0e-50aa-44d9-a9b1-49cdeec2fe36)
![Screenshot 2024-11-28 103712](https://github.com/user-attachments/assets/789d054f-e2a3-4aef-868a-a23ddd44de02)
![Screenshot 2024-11-28 103716](https://github.com/user-attachments/assets/52d406e0-1da4-44ad-83cb-62eba4a0be0c)
![Screenshot 2024-11-28 104224](https://github.com/user-attachments/assets/30b7448b-63c7-4d68-a40b-d277e7c2af8e)
![Screenshot 2024-11-28 103846](https://github.com/user-attachments/assets/81549c9a-4d3a-41da-ade8-a3f5a4343f7c)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
