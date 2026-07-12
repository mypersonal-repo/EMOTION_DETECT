import streamlit as st
import joblib
import pandas as pd

# Load saved models
logreg_model = joblib.load("logreg_model.pkl")
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Emoji mapping for emotions
emojis = {
    "joy": "😊",
    "anger": "😡",
    "sadness": "😢",
    "fear": "😨",
    "surprise": "😲",
    "love": "❤️",
    "neutral": "😐"
}

# Sidebar settings
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.radio("Choose Model:", ["Logistic Regression", "SVM", "Both"])

# Main UI
st.title("🎭 Emotion Detection Web App")
st.markdown("### 💬 Enter text below and get the predicted emotion!")

user_input = st.text_area("📝 Your text:", "")

if st.button("🔍 Predict Emotion"):
    if user_input.strip():
        features = vectorizer.transform([user_input])

        if model_choice in ["Logistic Regression", "Both"]:
            logreg_pred = logreg_model.predict(features)[0]
            logreg_probs = logreg_model.predict_proba(features)[0]

            st.subheader("Logistic Regression")
            st.write(f"Predicted Emotion: {emojis.get(logreg_pred, '🤔')} **{logreg_pred}**")
            st.bar_chart(pd.DataFrame({"Probability": logreg_probs}, index=logreg_model.classes_))

        if model_choice in ["SVM", "Both"]:
            svm_pred = svm_model.predict(features)[0]
            st.subheader("SVM")
            st.write(f"Predicted Emotion: {emojis.get(svm_pred, '🤔')} **{svm_pred}**")
    else:
        st.warning("⚠️ Please enter some text.")
