#streamlit used for creating user interface
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#joblib is used to save a model 
import joblib

#loading dataframe
df=pd.read_csv(r"C:\Users\Devika\Desktop\project\dataset.csv")

#assigning the independent variable
x=df.iloc[:,0:-1]

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(x)

#assigning the dependent/target variable
y=df.iloc[:,-1]

#splitting dataset in to test and train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#selecting the model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

#fiting the line 
model.fit(x_train,y_train)

#save the model
joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")

#UI For the streamlit app
st.title("CardioGuard-Protecting hearts with AI")
st.write("Enter the patient details below to predict heart disease.")
age=st.number_input("Age",min_value=1,max_value=120)
sex=st.selectbox("sex(0=female,1=male)",[0,1])
cp=st.selectbox("chest pain type(1=typical,2=typical anginal,3=anginal,4=assymptotic)",[1,2,3,4])
bp=st.number_input("resting bp s",min_value=0,max_value=200)
cholesterol=st.number_input("cholesterol",min_value=0)
fbp=st.selectbox("fasting blood sugar(1=>120,0=<120)",[1,0])
recg=st.selectbox("resting ecg(0=normal,1=abnormal,2=Borderline case)",[0,1,2])
mhr=st.number_input("max heart rate",min_value=60,max_value=202)
ex=st.selectbox("exercise angina(0=no,1=yes)",[0,1])
old=st.number_input("old speak")
sts=st.selectbox("ST slope(0=Normal,1=Upstoping,2=flat)",[0,1,2])



#predicting the output using user defined values

# Prediction button
if st.button("Predict"):
    # Prepare input data
    new_data=np.array([[age,sex,cp,bp,cholesterol,fbp,recg,mhr,ex,old,sts]])
    #input_data=scaler(new_data)
    #prediction takes here
    prediction = model.predict(new_data)

    # Display result
    result = "Heart Disease Detected 😟" if prediction[0] == 1 else "No Heart Disease 😊"
    st.subheader(result)





