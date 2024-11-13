# EX-01 Developing a Neural Network Regression Model
### Aim:
To develop a neural network regression model for the given dataset.

### Theory:
 - Design and implement a neural network regression model to accurately predict a continuous target variable based on a set of input features within the provided dataset. 
 - The objective is to develop a robust and reliable predictive model that can capture complex relationships in the data, ultimately yielding accurate and precise predictions of the target variable. 
 - The model should be trained, validated, and tested to ensure its generalization capabilities on unseen data, with an emphasis on optimizing performance metrics such as mean squared error or mean absolute error.

### Neural Network Model:
![image](https://github.com/user-attachments/assets/151f56b9-8129-4253-a9c3-744ab9c77732)

### Design Steps:

- STEP 1:Loading the dataset
- STEP 2:Split the dataset into training and testing
- STEP 3:Create MinMaxScalar objects ,fit the model and transform the data.
- STEP 4:Build the Neural Network Model and compile the model.
- STEP 5:Train the model with the training data.
- STEP 6:Plot the performance plot
- STEP 7:Evaluate the model with the testing data.

#### Name : SHARAN MJ
#### RegNo : 212222240097

## Program:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_excel('/content/owndata.xlsx')
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})
df

x=df[['input']].values
y=df[['output']].values
x

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
scalar=MinMaxScaler()
scalar.fit(x_train)

x_train1=scalar.transform(x_train)
ai=Sequential([
    Dense (units = 8, activation = 'relu'),
    Dense (units = 10, activation = 'relu'),
    Dense (units = 1)])

ai.compile(optimizer='rmsprop',loss='mse')
ai.fit(x_train1,y_train,epochs=2000)

loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()

X_test1 = scalar.transform(x_test)
ai.evaluate(X_test1,y_test)

X_n1 = [[float(input('enter the value : '))]]
X_n1_1 = scalar.transform(X_n1)
a=ai.predict(X_n1_1)
print('The predicted output : ',a)
```
### Output:

#### Dataset:
![Screenshot 2024-08-19 095045](https://github.com/user-attachments/assets/44b17a31-1662-4435-aa09-2c64557e55f7)

#### Training Loss Vs Iteration Plot:
![Screenshot 2024-08-19 095004](https://github.com/user-attachments/assets/fd9b0e7c-ccdf-4810-9d43-b002f19c8228)

#### Epoch:
![image](https://github.com/user-attachments/assets/3467338a-5505-4699-98a5-3875daa72c9d)


#### Test Data Root Mean Squared Error:
![image](https://github.com/user-attachments/assets/fb2766e9-8d79-413e-bef8-a94bdadb8a1f)

#### New Sample Data Prediction:
![image](https://github.com/user-attachments/assets/92dbd40a-7488-4aa3-b364-91311b99d591)

### Result:
Thus a neural network regression model for the given dataset is developed and the prediction for the given input is obtained accurately.
