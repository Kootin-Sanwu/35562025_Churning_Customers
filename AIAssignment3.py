import streamlit as st
import pandas as pd
import pickle
from keras.models import load_model

# Load the trained model
final_model_path = "final_model.h5"
final_model = load_model(final_model_path)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def preprocess_input(data):
    data_frame = pd.DataFrame(data, index=[0])
    
    # Replace empty strings with NaN and fill with mean for numeric columns
    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges", "customerID"]
    for columns in numeric_columns:
        data_frame[columns] = pd.to_numeric(data_frame[columns], errors='coerce')
        data_frame[columns].fillna(data_frame[columns].mean(), inplace=True)
        

    # Mapping categorical columns
    categorical_columns = {
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        "OnlineSecurity": ["Yes", "No"],
        "TechSupport": ["Yes", "No"],
        "OnlineBackup": ["Yes", "No"],
        "InternetService": ["DSL", "Fiber optic"],
    }
    
    ['tenure', 'MonthlyCharges', 'TotalCharges', 'customerID', 'Contract', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'OnlineBackup', 'InternetService']

    # Encode categorical columns
    label_encoders = {}
    for columns, values in categorical_columns.items():
        data_frame[columns] = data_frame[columns].apply(lambda x: values.index(x) if x in values else -1)  # Encode categorical columns
        label_encoders[columns] = values

    # Scale the data using the loaded scaler
    scaled_data = scaler.transform(data_frame)
    return scaled_data


# Function to make predictions
def predict_churn(data):
    scaled_data = preprocess_input(data)
    predictions = final_model.predict(scaled_data)
    churn_probability = predictions[0][0]

    if churn_probability > 0.5:
        prediction = "Churn"
        confidence = churn_probability
    else:
        prediction = "No Churn"
        confidence = 1 - churn_probability

    return prediction, confidence

# Streamlit app
def main():
    # Create input fields for user data
    st.title("Customer Churn Prediction")
    st.write("Enter Information for the new customer:")
    tenure = st.number_input("Tenure in Months:", min_value = 0)
    monthly_charges = st.number_input("Monthly Charges:", min_value = 0.0)
    total_charges = st.number_input("Total Charges:", min_value = 0.0)
    customerID = st.number_input("Customer ID: ", min_value = 0)
    contract = st.selectbox(
        "Contract:", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("PaymentMethod:", [
                              "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    online_security = st.selectbox("OnlineSecurity:", ["Yes", "No"])
    tech_support = st.selectbox("TechSupport:", ["Yes", "No"])
    online_backup = st.selectbox("OnlineBackup:", ["Yes", "No"])
    internet_service = st.selectbox("InternetService:", ["DSL", "Fiber optic"])
    

    user_input = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "customerID": [customerID],
        "Contract": [contract],
        "PaymentMethod": [payment_method],
        "OnlineSecurity": [online_security],
        "TechSupport": [tech_support],
        "OnlineBackup": [online_backup],
        "InternetService": [internet_service]
    })

    if st.button("Predict Churn"):
        prediction, confidence = predict_churn(user_input)

        st.write(f"Prediction: {prediction}, Confidence: {round(confidence * 100, 2)}%")

if __name__ == "__main__":
    main()