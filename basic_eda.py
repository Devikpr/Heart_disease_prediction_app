import numpy as np
import pandas as pd

# Ensure the file has a .csv extension
df = pd.read_csv(r"C:\Users\Devika\Desktop\project\dataset.csv")  

# Display the first few rows
print(df.head())  

#Familiarizing myself with the columns of the dataset.
print(df.columns)

#Familiarizing with the data types of the columns.
print(df.dtypes)

#Checking null values in the dataset
print(df.isnull().sum())

#Assignining dependent and independent variables.
x=df.iloc[:,:-1]
print(x.head())

y=df.iloc[:,-1]
print(y.head())

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

#fiting the the line
model.fit(X_train,Y_train)

#prediction using 

y_pred=model.predict(X_test)

#checking the accuracy
from sklearn.metrics import accuracy_score
print("Accuracy is",accuracy_score(Y_test,y_pred))

new_array=np.array([[40,1,2,140,289,0,0,172,0,0,1]])
new_pred=model.predict(new_array)
print(new_pred)
