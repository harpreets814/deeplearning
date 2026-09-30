import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle

# Load model
model = tf.keras.models.load_model("model.h5")

# Load gender encoder
with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load geography encoder
with open("onehot_encoder.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)


st.title("Customer Churn Prediction")

st.write("Enter customer details:")


# User inputs

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_of_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)


if st.button("Predict"):

    # Encode Gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Encode Geography
    geography_encoded = onehot_encoder.transform(
        [[geography]]
    ).toarray()[0]

    # Create input data
    input_data = pd.DataFrame([[
        credit_score,
        geography_encoded[0],
        geography_encoded[1],
        geography_encoded[2],
        gender_encoded,
        age,
        tenure,
        balance,
        num_of_products,
        has_cr_card,
        is_active_member,
        estimated_salary
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    probability = prediction[0][0]

    if probability >= 0.5:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is unlikely to churn")

    st.write(
        f"Churn probability: {probability:.2%}"
    )