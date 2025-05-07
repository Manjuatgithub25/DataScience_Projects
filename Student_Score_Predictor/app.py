import streamlit as st
import pandas as pd
import pickle 

with open("artifact/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("artifact/model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Student Performance Prediction")

st.sidebar.header("Input Features")

# Define input fields
def user_inputs():
    gender = st.sidebar.selectbox("gender", ["male", "female"])
    ethnicity = st.sidebar.selectbox("race_ethnicity", ['group B', 'group C', 'group A', 'group D', 'group E'])
    parent_education = st.sidebar.selectbox("parental_level_of_education", ["bachelor's degree", 'some college', "master's degree",
       "associate's degree", 'high school', 'some high school'])
    lunch_type = st.sidebar.selectbox("lunch", ['standard', 'free/reduced'])
    course_taken = st.sidebar.selectbox("test_preparation_course", ['none', 'completed'])
    reading_score = st.sidebar.slider("reading_score", 0, 100, 50)
    writing_score = st.sidebar.slider("writing_score", 0, 100, 50)

    return pd.DataFrame([[gender, ethnicity, parent_education, lunch_type, course_taken, reading_score, writing_score]], 
                        columns=["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course", "reading_score", "writing_score"])

input_df = user_inputs()

# Predict button
if st.button("Predict Performance"):
    transformed_input = preprocessor.transform(input_df)
    prediction = model.predict(transformed_input)
    st.success(f"Predicted Math Score: {prediction[0]:.2f}")
