import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# Load the trained model and optimal threshold
model = joblib.load("best_model__binaryRandomForest.joblib")  # Update the path if necessary
best_threshold = 0.5  # Replace with the optimal threshold you found during training

# Load the dataset to extract feature names and real data ranges
data = pd.read_csv("standarded_data_featureseng.csv")
feature_names = data.drop(columns=["Unnamed: 0", "num"]).columns.tolist()

# Initialize RobustScaler with the original data
scaler = RobustScaler()
scaler.fit(data[feature_names])

# Calculate real (non-standardized) min and max values for each feature
real_ranges = data[feature_names].agg(['min', 'max']).T

# App title and instructions
st.title("Heart Disease Prediction")
st.write("Please input the patient's details based on real-world values.")

# Helper function to display the real data range
def display_real_range(feature_name):
    min_val = real_ranges.loc[feature_name, 'min']
    max_val = real_ranges.loc[feature_name, 'max']
    st.markdown(f"**Expected real range for {feature_name}: [{min_val}, {max_val}]**")
    return min_val, max_val

# Create input fields for each feature based on real data ranges
input_data = {}

# Input fields with displayed real data ranges and robust scaling in the background
for feature in feature_names:
    min_val, max_val = display_real_range(feature)
    if feature in ["sex", "fbs", "restecg", "exang", "smoking", "diabetes", "age_above_50", "ecg_abnormality"]:
        # Binary features (0 or 1)
        input_data[feature] = st.number_input(f"Enter {feature} (0 or 1):", min_value=0.0, max_value=1.0, step=1.0)
    else:
        # Numerical features with real data ranges
        input_data[feature] = st.number_input(f"Enter {feature}:", min_value=min_val, max_value=max_val, step=0.01)

# Convert the inputs to a DataFrame
input_df = pd.DataFrame([input_data])

# Apply robust scaling to the input data
scaled_input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

# Predict button
if st.button("Predict"):
    # Predict probability using the scaled input data
    prob = model.predict_proba(scaled_input_df)[:, 1][0]
    
    # Apply the threshold to determine the prediction
    prediction = "Sick" if prob >= best_threshold else "Not Sick"
    
    # Display the result
    st.write(f"Prediction: {prediction}")
    st.write(f"Probability of being sick: {prob:.2%}")

    # Debugging output (optional)
    st.write("Scaled Input Data:", scaled_input_df)