import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("goemotions_prepared.csv")
X = df["text"]
y = df["emotion"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=20000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Train Logistic Regression
logreg_model = LogisticRegression(C=2.0, solver="saga", max_iter=2000, class_weight="balanced")
logreg_model.fit(X_resampled, y_resampled)

# Train SVM
svm_model = LinearSVC(class_weight="balanced", max_iter=2000)
svm_model.fit(X_resampled, y_resampled)

# Save models + vectorizer
joblib.dump(logreg_model, "logreg_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
