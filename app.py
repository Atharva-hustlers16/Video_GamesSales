# app.py

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and encoders
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load and clean data
df = pd.read_csv("vgsales.csv")
df_cleaned = df.dropna().copy()

# Encode cleaned data
df_cleaned['Genre'] = encoders['Genre'].transform(df_cleaned['Genre'])
df_cleaned['Platform'] = encoders['Platform'].transform(df_cleaned['Platform'])

# Add binary label for prediction
median_sales = df_cleaned['Global_Sales'].median()
df_cleaned['Global_Sales_High'] = (df_cleaned['Global_Sales'] > median_sales).astype(int)

# Model accuracy (recalculate using cleaned data)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df_cleaned[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Genre', 'Platform']]
y = df_cleaned['Global_Sales_High']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
accuracy = accuracy_score(y_test, model.predict(X_test)) * 100

# Streamlit App
st.title("🎮 Video Game Sales Predictor")

page = st.sidebar.selectbox("📑 Select Page", ["Raw Data", "Cleaned Data", "Prediction", "Data Visualization"])

# Page 1: Raw Data
if page == "Raw Data":
    st.subheader("🗂 Raw Data")
    st.dataframe(df)

# Page 2: Cleaned Data
elif page == "Cleaned Data":
    st.subheader("✅ Cleaned & Encoded Data")
    st.dataframe(df_cleaned.head(50))

# Page 3: Prediction
elif page == "Prediction":
    st.subheader("🔮 Predict Global Sales Category")

    na_sales = st.number_input("NA Sales", 0.0, 100.0, 1.0)
    eu_sales = st.number_input("EU Sales", 0.0, 100.0, 1.0)
    jp_sales = st.number_input("JP Sales", 0.0, 100.0, 1.0)
    genre = st.selectbox("Genre", encoders['Genre'].classes_)
    platform = st.selectbox("Platform", encoders['Platform'].classes_)

    if st.button("Predict"):
        genre_encoded = encoders['Genre'].transform([genre])[0]
        platform_encoded = encoders['Platform'].transform([platform])[0]
        input_data = [[na_sales, eu_sales, jp_sales, genre_encoded, platform_encoded]]

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        result = "High Sales" if prediction == 1 else "Low Sales"
        st.success(f"🎯 Predicted: {result}")

        st.write(f"✅ **Model Accuracy**: {accuracy:.2f}%")
        st.write(f"🔍 **Confidence**: High Sales = {probabilities[1]*100:.2f}%, Low Sales = {probabilities[0]*100:.2f}%")

# Page 4: Data Visualization
elif page == "Data Visualization":
    st.subheader("📊 Data Visualization (Cleaned Data)")

    fig1, ax1 = plt.subplots()
    sns.countplot(data=df_cleaned, x='Global_Sales_High', ax=ax1)
    ax1.set_title("Sales Class Distribution (0=Low, 1=High)")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df_cleaned, x='Global_Sales_High', y='NA_Sales', ax=ax2)
    ax2.set_title("NA Sales vs Sales Class")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.heatmap(df_cleaned[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Global_Sales_High']].corr(), annot=True, cmap='coolwarm', ax=ax3)
    ax3.set_title("Correlation Heatmap")
    st.pyplot(fig3)
