conda create -n streamlit_env python=3.11
conda activate streamlit_env
pip install streamlit numpy pandas pyarrow
import streamlit as st
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------
# Helper functions
# -------------------------------
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    return text

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📧 Spam Email Classifier")
st.write("A simple ML project using text preprocessing, feature extraction, and classification.")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file with 'label' and 'message' columns", type="csv")

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Preprocess text
    df['message'] = df['message'].apply(preprocess_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )

    # Feature extraction
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train classifier
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"### Model Accuracy: {acc:.2f}")
    st.write("### Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    # User input for prediction
    st.write("### Try it yourself")
    user_input = st.text_area("Enter an email message:")
    if user_input:
        processed = preprocess_text(user_input)
        vec = vectorizer.transform([processed])
        prediction = model.predict(vec)[0]
        st.write(f"Prediction: **{prediction}**")