# ML Loan Approval 
Jonathan Sebban David Grogan 

## Loan Approval Prediction

**Presenters:** David Grogan - Jonathan Sebban

Hello, this is my homework for ML. 

The Project have : 

- My ipynb for the EDA
- My ipynb where I did all the machine learning
- A py file name app.py where I did an application that you can try, where you can choose the variable and see if the loan will be approuved
- A file with the app and an api.
- the requirement file and the joblibs
- The data
  
Obviously you do what you want but you should install the requirement file and run the app.py file and try the model with the parameter you want. The app will tell you if the loan is approuved or denied and create a score.txt file with the result.


https://github.com/user-attachments/assets/eeb1d3e7-cb28-4529-bf13-2328dcf7681c


### Introduction

**Goal:** The objective is to predict whether an applicant is approved for a loan.

**Business Stakes:**
- Improve the accuracy of loan approval decisions.
- Reduce the risk of default.
- Maximize profitability by approving creditworthy applicants and minimizing losses from risky loans.

**Dataset Source:** The dataset was provided by Mr. Desforges in class and is part of a Kaggle competition.

**Evaluation:** Submissions are evaluated using area under the ROC curve with predicted probabilities and actual targets.

**Useful Links:**
- [Receiver Operating Characteristic on Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Loan Approval Prediction Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s4e10/overview)

### The DATA

**Attributes:**
- `id`: Unique identifier for each observation.
- `person_age`: Age of the loan applicant.
- `person_income`: Annual income of the applicant.
- `person_home_ownership`: Home ownership status of the applicant (Own, Rent, Mortgage).
- `person_emp_length`: Employment length of the applicant in years.
- `loan_intent`: Purpose of the loan (Educational, Medical, Personal, etc.).
- `loan_grade`: Loan rating from A to G indicating risk level.
- `loan_amnt`: Requested loan amount.
- `loan_int_rate`: Interest rate of the loan.
- `loan_percent_income`: Loan amount as a percentage of the applicant’s income.
- `cb_person_default_on_file`: Indicates previous loan defaults (Y/N).
- `cb_person_cred_hist_length`: Length of the applicant’s credit history in years.
- `loan_status`: Current status of the loan, used as the target variable in predictions.

### EDA

**Dataset Overview:**
- 12 columns
- 58,645 rows
- Described as relatively small dataset.

### Data Cleaning

**Initial State:** The dataset was already fairly clean upon receipt.

### Feature Engineering

The encoding was based on a success rate pipeline to see if there is any order in the value.

**Categorical Variables:** 
- `person_home_ownership`
- `loan_intent`
- `loan_grade`
- `cb_person_default_on_file`

**Encoding Decisions:**
- Applied OneHotEncoder for attributes without a natural order.
       `person_home_ownership`
       `loan_intent`
       `cb_person_default_on_file`
- Used Ordinal Encoder for `loan_grade` due to a clear order in variable impact.

### Outliers

**Adjustments:**

Encoding: Applied OneHotEncoder for categorical variables without order and Ordinal Encoder for loan_grade.
- Removed outliers in `person_age` beyond 100 years.
- <img width="574" alt="Screenshot 2024-11-08 at 14 15 02" src="https://github.com/user-attachments/assets/43ac33e6-3e6a-465e-81a2-2e5cf78b8045">

- Eliminated employment lengths over 60 years as unrealistic.
- <img width="583" alt="Screenshot 2024-11-08 at 14 15 31" src="https://github.com/user-attachments/assets/a90e7b5c-a99b-4efe-8ca5-dd15cce1c5b7">

  

### Modeling

**Baseline:**
- Utilized logistic regression with an ROC AUC of 0.88.

I tried every model to see wich one is the best. 

  ![Capture d'écran 2024-10-24 100318](https://github.com/user-attachments/assets/e17f6982-e484-4922-9b12-c19732d8a2ce)


**XGBoost Variants:**
- Classic XGBoost: ROC AUC of 0.9537.
- XGBoost with k-fold: Adjusted to avoid overfitting; slightly lower mean ROC AUC but improved score on the best fold.
- XGBoost with k-fold and added columns (`total_due_with_loan_amnt` and `remaining_income_end_of_month`): Resulted in a slight decrease in ROC AUC but columns were retained for their utility.

**Optimization with Optuna:**
- Enhanced model performance from 0.9519 to 0.9554 ROC AUC.
- Final test dataset score on Kaggle was 0.8771, suggesting potential overfitting on the training dataset.

### Conclusion

**Final Notes:**
- Potential strategies to improve future scores include reducing overfitting, refining or removing less effective features.

**Acknowledgements:**
- Special thanks to Mr. Desforges for guidance and data provision.
