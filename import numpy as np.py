import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv('creditcard.csv')

print(data.columns)
print(data.describe())

# Data preprocessing
data = data.sample(frac=0.1, random_state=1)
print(data.shape)

# Determine the number of fraud cases in the dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud) / float(len(valid))
print(outlier_fraction)
print("Fraud Cases: {}".format(len(fraud)))
print("Valid Cases: {}".format(len(valid)))

X = data.drop(['Class'], axis=1)
Y = data['Class']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)

print("Confusion Matrix: \n", confusion_matrix(Y_test, Y_pred))
print("Accuracy Score: ", accuracy_score(Y_test, Y_pred))
print("Classification Report: \n", classification_report(Y_test, Y_pred))
