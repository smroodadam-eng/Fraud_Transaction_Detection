import streamlit as st
import pandas as pd
import joblib

# Set page title
st.title("Fraud Detection System")

# Simple form for input
st.subheader("Enter Transaction Details")

# Get user inputs
type_txn = st.selectbox(
    "Transaction Type:",
    ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
)

amount = st.number_input("Amount:", min_value=0.0, value=1000.0)

st.write("**Origin Account:**")
oldbalanceOrg = st.number_input("Old Balance (Origin):", min_value=0.0, value=5000.0)
newbalanceOrig = st.number_input("New Balance (Origin):", min_value=0.0, value=4000.0)

st.write("**Destination Account:**")
oldbalanceDest = st.number_input("Old Balance (Destination):", min_value=0.0, value=1000.0)
newbalanceDest = st.number_input("New Balance (Destination):", min_value=0.0, value=2000.0)

# Predict button
if st.button("Check for Fraud"):
    try:
        # Load the model and files
        rf = joblib.load('rf_model.pkl')
        le = joblib.load('type_encoder.pkl')
        amount_bins = joblib.load('amount_bins.pkl')

        # Create transaction data
        new_txn = pd.DataFrame([{
            'type': type_txn,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest
        }])

        # Prepare features (same as original code)
        new_txn['type_code'] = le.transform(new_txn['type'])
        new_txn['orig_diff'] = new_txn['oldbalanceOrg'] - new_txn['newbalanceOrig']
        new_txn['dest_diff'] = new_txn['newbalanceDest'] - new_txn['oldbalanceDest']
        new_txn['balance_error'] = (new_txn['oldbalanceOrg'] - new_txn['amount']) - new_txn['newbalanceOrig']
        new_txn['neg_balance_orig'] = (new_txn['oldbalanceOrg'] - new_txn['amount']) < 0

        new_txn['amount_q'] = pd.cut(
            new_txn['amount'],
            bins=amount_bins,
            include_lowest=True
        )
        new_txn['amount_q_code'] = new_txn['amount_q'].cat.codes

        # Select features for prediction
        X_new = new_txn[['amount', 'amount_q_code', 'type_code',
                         'orig_diff', 'dest_diff', 'balance_error',
                         'neg_balance_orig']]

        # Make prediction
        is_fraud = rf.predict(X_new)[0]
        fraud_prob = rf.predict_proba(X_new)[0, 1]

        # Show results
        st.subheader("Prediction Results")

        if is_fraud:
            st.error("FRAUD DETECTED")
        else:
            st.success("NOT FRAUD")

        st.write(f"Fraud Probability: {fraud_prob:.2%}")

        # Show transaction summary
        st.subheader("Transaction Summary")
        st.write(f"Type: {type_txn}")
        st.write(f"Amount: ${amount:,.2f}")
        st.write(f"Origin: ${oldbalanceOrg:,.2f} → ${newbalanceOrig:,.2f}")
        st.write(f"Destination: ${oldbalanceDest:,.2f} → ${newbalanceDest:,.2f}")

    except FileNotFoundError:
        st.error("Error: Model files not found.")
        st.write("Please make sure these files are in the same folder:")
        st.write("- rf_model.pkl")
        st.write("- type_encoder.pkl")
        st.write("- amount_bins.pkl")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Simple instructions
st.write("---")
st.write("**Instructions:**")
st.write("1. Fill in all the transaction details above")
st.write("2. Click 'Check for Fraud' button")
st.write("3. View the prediction results")