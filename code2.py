import pandas as pd
import numpy as np
import joblib

# Load the model and encoders
rf = joblib.load('rf_model.pkl')
le = joblib.load('type_encoder.pkl')
amount_bins = joblib.load('amount_bins.pkl')




# Ask user for transaction details
type_txn = input("Enter transaction type (TRANSFER, CASH_OUT, PAYMENT, etc.): ")
amount = float(input("Enter transaction amount: "))
oldbalanceOrg = float(input("Enter old balance of origin account: "))
newbalanceOrig = float(input("Enter new balance of origin account: "))
oldbalanceDest = float(input("Enter old balance of destination account: "))
newbalanceDest = float(input("Enter new balance of destination account: "))

# Put into DataFrame
new_txn = pd.DataFrame([{
    'type': type_txn,
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest
}])

# Feature engineering
new_txn['type_code'] = le.transform(new_txn['type'])
new_txn['orig_diff'] = new_txn['oldbalanceOrg'] - new_txn['newbalanceOrig']
new_txn['dest_diff'] = new_txn['newbalanceDest'] - new_txn['oldbalanceDest']
new_txn['balance_error'] = (new_txn['oldbalanceOrg'] - new_txn['amount']) - new_txn['newbalanceOrig']
new_txn['neg_balance_orig'] = (new_txn['oldbalanceOrg'] - new_txn['amount']) < 0

# Amount quantile code using saved bins
new_txn['amount_q'] = pd.cut(
    new_txn['amount'],
    bins=amount_bins,
    include_lowest=True
)
new_txn['amount_q_code'] = new_txn['amount_q'].cat.codes

# Select features
X_new = new_txn[['amount', 'amount_q_code', 'type_code', 'orig_diff', 'dest_diff', 'balance_error', 'neg_balance_orig']]

# Predict fraud
is_fraud = rf.predict(X_new)[0]
fraud_prob = rf.predict_proba(X_new)[0,1]

print("\n=== Fraud Prediction ===")
print("Is Fraud?", "Yes" if is_fraud else "No")
print(f"Probability of Fraud: {fraud_prob:.2f}")
