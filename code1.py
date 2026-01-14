import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv(r"C:\Users\smroo\OneDrive\Desktop\samrood_projects\Fraud_Detection\AIML Dataset.csv")
print(df.head())
print(df.info())
print(df.columns)
print(df['isFraud'].value_counts())
print(df['isFlaggedFraud'].value_counts())
print(df.shape)
print(df.isnull().sum())
print(df['isFraud'].value_counts()[1] / df.shape[0] * 100)
# Fraud distribution by transaction type
sns.countplot(data=df, x='type', hue='isFraud')
plt.title("Fraud Distribution by Transaction Type")
plt.show()
# Fraud vs Non-Fraud Amount distribution
sns.boxplot(data=df, x='isFraud', y='amount')
plt.title("Amount Distribution by Fraud Status")
plt.show()
# Correlation heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# feature engineering
df2 = df.loc[df['isFraud'] == 1]
print(df2.groupby('type')['isFraud'].sum().reset_index(name='type_count').sort_values(by='type_count'))
print(df.groupby('type')['isFraud'].mean().reset_index(name='AVG fraud rate').sort_values(by='AVG fraud rate'))
print(df.groupby('amount'))
df['amount_q'] = pd.qcut(df['amount'], q=8)

# SAVE bins
amount_bins = df['amount_q'].cat.categories
joblib.dump(amount_bins, 'amount_bins.pkl')
df['amount_q_code'] = df['amount_q'].cat.codes
print(df.groupby('amount_q')['amount'].agg(min_amt='min', max_amt='max').reset_index())
print(df.groupby('amount_q')['isFraud'].mean().reset_index(name='q_mean').sort_values(by='amount_q'))

df['orig_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['dest_diff'] = df['newbalanceDest'] - df['oldbalanceDest']
df['balance_error'] = (df['oldbalanceOrg'] - df['amount']) - df['newbalanceOrig']
print(df.loc[df['balance_error'] != 0])
print(df['balance_error'].max())
print(df['balance_error'].min())
print(df.groupby(df['balance_error'] != 0)['isFraud'].mean())
print(df.groupby(df['balance_error'] < 0)['isFraud'].mean())
df['neg_balance_orig'] = (df['oldbalanceOrg'] - df['amount']) < 0
print(df['neg_balance_orig'].value_counts())
print(df.groupby('neg_balance_orig')['isFraud'].mean() * 100)
print(df.groupby('isFraud')[['orig_diff', 'balance_error', 'amount']].mean())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['type_code'] = le.fit_transform(df['type'])
# Convert 'amount_q' intervals to numeric codes
df['amount_q_code'] = df['amount_q'].cat.codes
# Select features
X = df[['amount', 'amount_q_code', 'type_code', 'orig_diff', 'dest_diff', 'balance_error', 'neg_balance_orig']]
y = df['isFraud']
joblib.dump(le, 'type_encoder.pkl')
# training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# RF is used for model prediction
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
joblib.dump(rf, 'rf_model.pkl')
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# ROC AUC score
roc_auc = roc_auc_score(y_test, y_prob)
print("\nROC AUC Score:", roc_auc)
