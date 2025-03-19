# EX 03-Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:PRAVEEN.K
RegisterNumber:212223040152  
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
#### data information:
![Screenshot 2025-03-19 014304](https://github.com/user-attachments/assets/4b8e5858-195c-4ad4-995c-9bf09777acc9)


#### Value of x:
![Screenshot 2025-03-19 014322](https://github.com/user-attachments/assets/6677b4de-91b1-4343-8a4e-66a58f78b05f)

#### Value of X1_scaled:
![Screenshot 2025-03-19 014408](https://github.com/user-attachments/assets/62ea40a6-1548-4379-b30d-6d19287af834)

#### predicted value:

![Screenshot 2025-03-19 014416](https://github.com/user-attachments/assets/d8a3ed65-ee23-43bb-a336-0215f4d3c249)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
