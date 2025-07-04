import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pickle
import streamlit as st


# Load the pickle files and trained model
model = load_model("model.h5")                              # load model

with open("labelEncoding_gender.pkl", "rb") as file:        # Load Label Encoder
    label_encoder_gender = pickle.load(file)

with open("OneHotEncoding_geo.pkl", "rb") as file:          # Load one hot encoder
    one_hot_encoder_geo = pickle.load(file)

with open("Scaler.pkl", "rb") as file:                      # Load standard Scaler
    scaler = pickle.load(file)


# Streamlit app

st.title("Customer Churn Prediction")

# User input section
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# oreoare input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(["Geography"]))

input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_scaled = scaler.transform(input_df)

prediction_probability = model.predict(input_scaled)[0][0]
prediction_probability = float(prediction_probability)
st.write("The probability of getting churned: ", prediction_probability)

if prediction_probability < 0.5:
    st.write("Customer more likely to not leave")
else:
    st.write("Customer more likely to leave")