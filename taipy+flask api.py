import threading
from flask import Flask, request, jsonify
from taipy import Gui
import pandas as pd
import joblib
import xgboost as xgb
import os

# Configuration de Flask
app = Flask(__name__)

# Charger les modèles et les encoders comme dans votre script
base_dir = os.path.dirname(os.path.abspath(__file__))
ord_enc = joblib.load(os.path.join(base_dir, 'ordinal_encoder.joblib'))
ohe = joblib.load(os.path.join(base_dir, 'onehot_encoder.joblib'))
scaler = joblib.load(os.path.join(base_dir, 'scaler.joblib'))
feature_columns = joblib.load(os.path.join(base_dir, 'feature_columns.joblib'))
ohe_feature_names = joblib.load(os.path.join(base_dir, 'ohe_feature_names.joblib'))
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(base_dir, 'xgb_model.json'))

# Endpoint Flask pour prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON de la requête
        data = request.get_json()
        input_data = pd.DataFrame([data])

        # Transformations de données similaires à `make_prediction`
        input_data['loan_to_income_ratio'] = input_data['loan_amnt'] / input_data['person_income']
        input_data['total_due_with_loan_amnt'] = (
            input_data['loan_amnt'] * (input_data['loan_int_rate'] / 100)
        ) + input_data['loan_amnt']
        input_data['remaining_income_end_of_month'] = (
            input_data['person_income'] - input_data['total_due_with_loan_amnt']
        ) / 12
        input_data['loan_grade'] = ord_enc.transform(input_data[['loan_grade']])

        ohe_columns = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
        ohe_encoded = ohe.transform(input_data[ohe_columns])
        ohe_encoded_df = pd.DataFrame(
            ohe_encoded, index=input_data.index, columns=ohe_feature_names
        )

        input_data = input_data.drop(columns=ohe_columns)
        input_data = pd.concat([input_data, ohe_encoded_df], axis=1)

        numerical_columns = [
            'person_age', 'person_income', 'person_emp_length', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'loan_to_income_ratio',
            'total_due_with_loan_amnt', 'remaining_income_end_of_month'
        ]
        input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
        
        X_input = input_data[feature_columns]
        prediction = xgb_model.predict(X_input)
        result = 'Approved' if prediction[0] == 0 else 'Denied'

        return jsonify({'prediction_result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Configuration de Taipy
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

def make_prediction(state):
    # Fonction pour faire des prédictions en interne
    data = {
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
    }
    # Appel de l'API Flask pour obtenir le résultat de la prédiction
    try:
        response = app.test_client().post('/predict', json=data)
        if response.status_code == 200:
            result = response.json['prediction_result']
            state.prediction_result = result
        else:
            state.prediction_result = "Erreur lors de la prédiction"
    except Exception as e:
        state.prediction_result = f"Erreur: {str(e)}"

page = """
# Loan Approval Prediction

## Entrer les informations ci-dessous pour prédire l'approbation du prêt:

### Âge
<|{person_age}|number|label=Âge|>

### Revenu annuel
<|{person_income}|number|label=Revenu annuel|>

### Années d'emploi
<|{person_emp_length}|number|label=Années d'emploi|>

### Montant du prêt
<|{loan_amnt}|number|label=Montant du prêt|>

### Taux d'intérêt (%)
<|{loan_int_rate}|number|label=Taux d'intérêt (%)|>

### Pourcentage du revenu pour le prêt
<|{loan_percent_income}|number|label=Pourcentage du revenu pour le prêt|>

### Grade du prêt
<|{loan_grade}|selector|lov=A;B;C;D;E;F;G|dropdown|label=Grade du prêt|>

### Propriété de la maison
<|{person_home_ownership}|selector|lov=RENT;OWN;MORTGAGE;OTHER|dropdown|label=Propriété de la maison|>

### Intention de prêt
<|{loan_intent}|selector|lov=PERSONAL;EDUCATION;MEDICAL;VENTURE;HOMEIMPROVEMENT;DEBTCONSOLIDATION|dropdown|label=Intention de prêt|>

### Défaut enregistré
<|{cb_person_default_on_file}|selector|lov=Y;N|dropdown|label=Défaut enregistré|>

## Prédiction
<|Faire une prédiction|button|on_action=make_prediction|class=btn btn-primary|>

### Résultat:
<|{prediction_result}|>
"""

gui = Gui(page=page)

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

# Lancer Flask et Taipy sur deux threads distincts
def run_flask():
    app.run(port=5000)

def run_taipy():
    gui.run(port=5001)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    taipy_thread = threading.Thread(target=run_taipy)

    flask_thread.start()
    taipy_thread.start()
