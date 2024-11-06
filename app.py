# taipy_app.py

from taipy import Gui
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
import traceback

# Get the absolute path to the directory containing this script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load necessary encoders and scaler using absolute paths
ord_enc = joblib.load(os.path.join(base_dir, 'ordinal_encoder.joblib'))
ohe = joblib.load(os.path.join(base_dir, 'onehot_encoder.joblib'))
scaler = joblib.load(os.path.join(base_dir, 'scaler.joblib'))

# Load the feature columns and OneHotEncoder feature names
feature_columns = joblib.load(os.path.join(base_dir, 'feature_columns.joblib'))
ohe_feature_names = joblib.load(os.path.join(base_dir, 'ohe_feature_names.joblib'))

# Load the trained XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(base_dir, 'xgb_model.json'))

# **Define variables in the module's scope (not passed to Gui constructor)**
person_age = 30
person_income = 50000.0
person_emp_length = 5.0
loan_amnt = 10000.0
loan_int_rate = 5.0
loan_percent_income = 0.2
loan_grade = 'B'
person_home_ownership = 'RENT'
loan_intent = 'PERSONAL'
cb_person_default_on_file = 'N'
prediction_result = ''

# **Define the make_prediction function before creating the GUI**
def make_prediction(state):
    try:
        print("make_prediction function called")
        # Create a DataFrame with the input data from the state object
        input_data = pd.DataFrame([{
            'person_age': state.person_age,
            'person_income': state.person_income,
            'person_emp_length': state.person_emp_length,
            'loan_amnt': state.loan_amnt,
            'loan_int_rate': state.loan_int_rate,
            'loan_percent_income': state.loan_percent_income,
            'loan_grade': state.loan_grade,
            'person_home_ownership': state.person_home_ownership,
            'loan_intent': state.loan_intent,
            'cb_person_default_on_file': state.cb_person_default_on_file
        }])

        # Feature Engineering
        input_data['loan_to_income_ratio'] = input_data['loan_amnt'] / input_data['person_income']
        input_data['total_due_with_loan_amnt'] = (
            input_data['loan_amnt'] * (input_data['loan_int_rate'] / 100)
        ) + input_data['loan_amnt']
        input_data['remaining_income_end_of_month'] = (
            input_data['person_income'] - input_data['total_due_with_loan_amnt']
        ) / 12

        # Convert 'loan_grade' to numeric using the saved OrdinalEncoder
        input_data['loan_grade'] = ord_enc.transform(input_data[['loan_grade']])

        # One-hot encode other categorical columns using the saved OneHotEncoder
        ohe_columns = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
        ohe_encoded = ohe.transform(input_data[ohe_columns])
        ohe_encoded_df = pd.DataFrame(
            ohe_encoded,
            index=input_data.index,
            columns=ohe_feature_names
        )

        # Drop original categorical columns and concatenate the encoded columns
        input_data = input_data.drop(columns=ohe_columns)
        input_data = pd.concat([input_data, ohe_encoded_df], axis=1)

        # Scale numerical features using the saved scaler
        numerical_columns = [
            'person_age', 'person_income', 'person_emp_length', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'loan_to_income_ratio',
            'total_due_with_loan_amnt', 'remaining_income_end_of_month'
        ]
        input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

        # Reorder columns to match the model's expected feature order
        X_input = input_data[feature_columns]

        # Make prediction
        prediction = xgb_model.predict(X_input)
        
        # Print the prediction result
        print(f"Prediction: {'Approved' if prediction[0] == 0 else 'Denied'}")

        # Map the prediction to a readable format
        state.prediction_result = 'Approved' if prediction[0] == 0 else 'Denied'
        print(f"state.prediction_result set to: {state.prediction_result}")

    except Exception as e:
        traceback.print_exc()
        state.prediction_result = f"Error: {str(e)}"
        print(f"Error occurred: {state.prediction_result}")


# **Define the Taipy GUI layout**
page = """
# Loan Approval Prediction

Please enter the following details to predict loan approval:

## Applicant Information

### Age
(Minimum age is 18)
<|{person_age}|number|label=Age|>

### Annual Income
(Must be positive)
<|{person_income}|number|label=Annual Income|>

### Years of Employment
(Zero or positive)
<|{person_emp_length}|number|label=Years of Employment|>

## Loan Information

### Loan Amount
(Must be positive)
<|{loan_amnt}|number|label=Loan Amount|>

### Interest Rate (%)
(Must be positive)
<|{loan_int_rate}|number|label=Interest Rate (%)|>

### Loan Percent Income
(Must be positive)
<|{loan_percent_income}|number|label=Loan Percent Income|>

### Loan Grade
(Select a grade from A to G)
<|{loan_grade}|selector|lov=A;B;C;D;E;F;G|dropdown|label=Loan Grade|>


### Loan Intent
(Select the purpose of the loan)
<|{loan_intent}|selector|lov=PERSONAL;EDUCATION;MEDICAL;VENTURE;HOMEIMPROVEMENT;DEBTCONSOLIDATION|dropdown|label=Loan Intent|>


### Home Ownership
(Select from RENT, OWN, MORTGAGE, OTHER)
<|{person_home_ownership}|selector|lov=RENT;OWN;MORTGAGE;OTHER|dropdown|label=Home Ownership|>

### Default on File
(Select 'Y' or 'N')
<|{cb_person_default_on_file}|selector|lov=Y;N|dropdown|label=Default on File|>

## Prediction

<|Make Prediction|button|on_action=make_prediction|class=btn btn-primary|>

### Result:
<|{prediction_result}|>
"""

# **Create the Taipy GUI without passing variables as arguments**
gui = Gui(page=page)

# **Initialize variables in the state using on_init**
def on_init(state):
    state.person_age = person_age
    state.person_income = person_income
    state.person_emp_length = person_emp_length
    state.loan_amnt = loan_amnt
    state.loan_int_rate = loan_int_rate
    state.loan_percent_income = loan_percent_income
    state.loan_grade = loan_grade
    state.person_home_ownership = person_home_ownership
    state.loan_intent = loan_intent
    state.cb_person_default_on_file = cb_person_default_on_file
    state.prediction_result = prediction_result

gui.on_init = on_init

# **Run the application on port 5001**
gui.run(port=5001)
