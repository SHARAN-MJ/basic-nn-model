# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: SHARAN MJ
### Register Number:212222240097

### DEPENDENCIES
```python



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


```
### DATA FROM GSHEETS
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('exp1').sheet1

rows = worksheet.get_all_values()
```
### DATA PROCESSING
```
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})
df.head()

x=df[['input']].values
y=df[['output']].values
x


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)

scalar=MinMaxScaler()

scalar.fit(x_train)

x_train1=scalar.transform(x_train)

```

### MODEL ARCHITECTURE AND TRAINING
```
 ai=Sequential([
    Dense (units = 8, activation = 'relu'),
    Dense (units = 10, activation = 'relu'),
    Dense (units = 1)])

ai.compile(optimizer='rmsprop',loss='mse')
ai.fit(x_train1,y_train,epochs=2000)
```
### LOSS CALCULATION

```
loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()
```

### PREDICTION
```
X_test1 = scalar.transform(x_test)
ai.evaluate(X_test1,y_test)

X_n1 = [[input("ENTER THE INPUT:")]]
X_n1_1 = scalar.transform(X_n1)
ai.predict(X_n1_1)
```

## Dataset Information
![Screenshot 2024-08-19 151204](https://github.com/user-attachments/assets/9caf09c2-ee46-43f2-b64d-8b8ddea04356)



## OUTPUT

### Training Loss Vs Iteration Plot
![Screenshot 2024-08-19 151236](https://github.com/user-attachments/assets/cc6905ad-aa76-49c7-9020-db4571fa9b78)


### Test Data Root Mean Squared Error
![Screenshot 2024-08-19 153301](https://github.com/user-attachments/assets/4da39e9f-f197-46e1-bddc-3038e03cd4b7)


### New Sample Data Prediction
![Screenshot 2024-08-19 153255](https://github.com/user-attachments/assets/71b2fc35-0cfc-41a7-8881-0f2e239bf4d2)


## RESULT

Include your result here
