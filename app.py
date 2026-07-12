import streamlit as st
import joblib
import pandas as pd

# Load saved models
logreg_model = joblib.load("logreg_model.pkl")
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# UI
st.title("🎭 Emotion Detection Web App")
st.write("Enter text below and get the predicted emotion!")

user_input = st.text_area("Your text:", "")

if st.button("Predict Emotion"):
    if user_input.strip():
        features = vectorizer.transform([user_input])
        
        # Logistic Regression
        logreg_pred = logreg_model.predict(features)[0]
        logreg_probs = logreg_model.predict_proba(features)[0]
        
        st.subheader("Logistic Regression")
        st.write(f"Predicted Emotion: **{logreg_pred}**")
        st.bar_chart(pd.DataFrame({"Probability": logreg_probs}, index=logreg_model.classes_))
        
        # SVM
        svm_pred = svm_model.predict(features)[0]
        st.subheader("SVM")
        st.write(f"Predicted Emotion: **{svm_pred}**")
    else:
        st.warning("Please enter some text.")
