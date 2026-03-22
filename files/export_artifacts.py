"""
Run this script ONCE after training your model in the notebook.
It saves the scaler and feature column names needed by the API.

Usage:
    python export_artifacts.py

Make sure lending_club_model.keras is already saved (model.save(...) in notebook).
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

print("Loading and preprocessing data...")

data0 = pd.read_csv("lending_club_loan_two.csv")
data1 = data0.copy()
data1.loan_status = pd.get_dummies(data1.loan_status, drop_first=True)

# --- Replicate full notebook pipeline ---
data2 = data1.copy()
data2.earliest_cr_line = pd.to_datetime(data2.earliest_cr_line)
data2["earliest_cr_line_month"] = data2.earliest_cr_line.apply(lambda x: x.month)
data2["earliest_cr_line_year"] = data2.earliest_cr_line.apply(lambda x: x.year)
data2.drop("earliest_cr_line", axis=1, inplace=True)

data3 = data2.copy()
data3.sub_grade = data3.sub_grade.apply(lambda x: x[1])

data4 = data3.copy()
data4.address = data4.address.apply(lambda x: x[-5:])

data5 = data4.copy()
data5.drop(
    ["emp_title", "title", "issue_d", "earliest_cr_line_month",
     "pub_rec_bankruptcies", "initial_list_status", "emp_length", "sub_grade"],
    axis=1, inplace=True
)

fill_df = data5.groupby("total_acc").mort_acc.mean()

def fill_mort_acc(tot, mort):
    if np.isnan(mort):
        return fill_df[tot]
    return mort

data5.mort_acc = data5.apply(lambda x: fill_mort_acc(x.total_acc, x.mort_acc), axis=1)
data5.dropna(inplace=True)
data5.home_ownership = data5.home_ownership.replace(["NONE", "ANY"], "OTHER")
data5.term = data5.term.apply(lambda x: int(x[1:3]))

dummies = pd.get_dummies(data5.select_dtypes("str"), drop_first=True)
data5.drop(data5.select_dtypes("str").columns.to_list(), axis=1, inplace=True)
data11 = pd.concat([data5, dummies], axis=1)

X = data11.drop("loan_status", axis=1)
feature_columns = X.columns.tolist()
X_vals = X.values
y = data11.loan_status.values

X_train, X_test, y_train, y_test = train_test_split(X_vals, y, test_size=0.33, random_state=42)

scaler = MinMaxScaler()
scaler.fit_transform(X_train)

# Save artifacts
artifacts = {
    "scaler": scaler,
    "feature_columns": feature_columns,
}
with open("scaler.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print(f"✅ Saved scaler.pkl")
print(f"✅ Feature columns ({len(feature_columns)}): {feature_columns}")
print("\nNow you can build the Docker image.")
