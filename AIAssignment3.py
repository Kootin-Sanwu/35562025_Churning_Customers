import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import load_model
import pickle
import numpy as np

# Load the trained model
final_model_path = "final_model.h5"
final_model = load_model(final_model_path)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Customer Churn Prediction")
# Create input fields for user data
st.write("Enter Information for the new customer:")
tenure = st.number_input("Tenure in Months:", min_value=0)
monthly_charges = st.number_input("Monthly Charges:", min_value=0.0)
total_charges = st.number_input("Total Charges:", min_value=0.0)
online_backup = st.selectbox("OnlineBackup:", ["Yes", "No"])
gender = st.selectbox("Gender:", ["Male", "Female"])
partner = st.selectbox("Partner:", ["Yes", "No"])
multiple_lines = st.selectbox(
    "MultipleLines:", ["Yes", "No phone service", "No"])
internet_service = st.selectbox("InternetService:", ["DSL", "Fiber optic"])
online_security = st.selectbox("OnlineSecurity:", ["Yes", "No"])
device_protection = st.selectbox("DeviceProtection:", ["Yes", "No"])
tech_support = st.selectbox("TechSupport:", ["Yes", "No"])
contract = st.selectbox(
    "Contract:", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("PaperlessBilling:", ["Yes", "No"])
payment_method = st.selectbox("PaymentMethod:", [
                              "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

user_input = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "OnlineBackup": [online_backup],
    "gender": [gender],
    "Partner": [partner],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method]
})

label_encoders = {}
categorical_features = ["OnlineBackup", "gender", "Partner", "OnlineSecurity",
                        "DeviceProtection", "TechSupport", "Contract", "PaperlessBilling", "PaymentMethod"]
for feature in categorical_features:
    le = LabelEncoder()
    le.fit(user_input[feature])
    label_encoders[feature] = le

for feature in categorical_features:
    user_input[feature] = label_encoders[feature].transform(
        user_input[feature])

if st.button("Predict"):
    input_data = pd.DataFrame(user_input, index=[0])
    input_data.replace(" ", np.nan, inplace=True)
    input_data.fillna(input_data.mean(), inplace=True)
    input_data = scaler.transform(input_data)
    prediction = final_model.predict(input_data)
    churn_probability = prediction[0][0]

    st.write(f"Churn Probability: {churn_probability:.4f}")

    if churn_probability > 0.5000:
        st.write("This customer is likely to churn")
    else:
        st.write("This customer is not likely to churn")


# import streamlit as st
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from keras.models import load_model
# import pickle
# import numpy as np

# # Load the trained model
# final_model_path = "final_model.h5"
# final_model = load_model(final_model_path)

# with open("scaler.pkl", "rb") as scaler_file:
#     scaler = pickle.load(scaler_file)

# st.title("Customer Churn Prediction")
# # Create input fields for user data
# st.write("Enter Information for the new customer:")
# tenure = st.number_input("Tenure in Months:", min_value=0)
# monthly_charges = st.number_input("Monthly Charges:", min_value=0.0)
# total_charges = st.number_input("Total Charges:", min_value=0.0)
# online_backup = st.selectbox("OnlineBackup:", ["Yes", "No"])
# gender = st.selectbox("Gender:", ["Male", "Female"])
# partner = st.selectbox("Partner:", ["Yes", "No"])
# multiple_lines = st.selectbox("MultipleLines:", ["Yes", "No phone service", "No"])
# internet_service = st.selectbox("InternetService:", ["DSL", "Fiber optic"])
# online_security = st.selectbox("OnlineSecurity:", ["Yes", "No"])
# device_protection = st.selectbox("DeviceProtection:", ["Yes", "No"])
# tech_support = st.selectbox("TechSupport:", ["Yes", "No"])
# contract = st.selectbox("Contract:", ["Month-to-month", "One year", "Two year"])
# paperless_billing = st.selectbox("PaperlessBilling:", ["Yes", "No"])
# payment_method = st.selectbox("PaymentMethod:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# user_input = pd.DataFrame({
#     "tenure": [tenure],
#     "MonthlyCharges": [monthly_charges],
#     "TotalCharges": [total_charges],
#     "OnlineBackup": [online_backup],
#     "gender": [gender],
#     "Partner": [partner],
#     "MultipleLines": [multiple_lines],
#     "InternetService": [internet_service],
#     "OnlineSecurity": [online_security],
#     "DeviceProtection": [device_protection],
#     "TechSupport": [tech_support],
#     "Contract": [contract],
#     "PaperlessBilling": [paperless_billing],
#     "PaymentMethod": [payment_method]
# })

# # Load the label encoders
# with open("label_encoders.pkl", "rb") as label_encoders_file:
#     label_encoders = pickle.load(label_encoders_file)

# categorical_features = ["OnlineBackup", "gender", "Partner", "OnlineSecurity", "DeviceProtection", "TechSupport", "Contract", "PaperlessBilling", "PaymentMethod"]

# for feature in categorical_features:
#     # Transform categorical columns using one-hot encoding
#     user_input = pd.concat([user_input, pd.get_dummies(user_input[feature], prefix=feature)], axis=1)
#     # Drop the original categorical column
#     user_input.drop(feature, axis=1, inplace=True)

# # Ensure all columns are present
# missing_columns = set(label_encoders.keys()) - set(user_input.columns)
# for column in missing_columns:
#     user_input[column] = 0

# # Reorder columns to match the trained model
# user_input = user_input[label_encoders.keys()]

# if st.button("Predict"):
#     input_data = pd.DataFrame(user_input, index=[0])
#     input_data.replace(" ", np.nan, inplace=True)
#     input_data.fillna(input_data.mean(), inplace=True)
#     input_data = scaler.transform(input_data)
#     prediction = final_model.predict(input_data)
#     churn_probability = prediction[0][0]

#     st.write(f"Churn Probability: {churn_probability:.4f}")

#     if churn_probability > 0.5000:
#         st.write("This customer is likely to churn")
#     else:
#         st.write("This customer is not likely to churn")
