import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Data
@st.cache
def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train, test

train, test = load_data()

# Data Exploration
st.title("Titanic - Machine Learning from Disaster")
st.write("Predicting the survival of Titanic passengers using Machine Learning.")

if st.checkbox("Show raw training data"):
    st.write(train.head())

# Data Preprocessing
st.header("Data Preprocessing")

# Fill missing values
imputer = SimpleImputer(strategy='median')
train['Age'] = imputer.fit_transform(train[['Age']])
test['Age'] = imputer.transform(test[['Age']])

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

train.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
test.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

# Encode categorical features
encoder = LabelEncoder()
for col in ['Sex', 'Embarked']:
    train[col] = encoder.fit_transform(train[col])
    test[col] = encoder.transform(test[col])

# Feature and target selection
X = train.drop(['Survived', 'PassengerId'], axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
st.header("Model Training")
st.write("Training a RandomForestClassifier")

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Feature Importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)
st.bar_chart(feature_importance.set_index('Feature'))

# Prediction on User Input
st.header("Survival Prediction")
st.write("Enter passenger details to predict survival.")

def user_input_features():
    pclass = st.selectbox("Ticket Class", [1, 2, 3])
    sex = st.selectbox("Sex", ['Male', 'Female'])
    age = st.slider("Age", 0, 100, 25)
    sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.slider("Parents/Children Aboard", 0, 6, 0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
    embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])
    
    sex = 1 if sex == 'Male' else 0
    embarked = {'C': 0, 'Q': 1, 'S': 2}[embarked]
    
    data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }
    return pd.DataFrame([data])

input_df = user_input_features()
prediction = clf.predict(input_df)[0]
st.write("Prediction:", "Survived" if prediction == 1 else "Did Not Survive")

# Run Streamlit App
if __name__ == "__main__":
    st.write("End of Streamlit Titanic Project")