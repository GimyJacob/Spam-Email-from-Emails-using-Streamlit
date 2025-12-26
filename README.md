spam-email-classifier/
spam_model.pkl        # Pre-trained ML model
vectorizer.pkl        # Pre-trained text vectorizer
app.py                # Streamlit application code
README.md             # Project documentation
 Features
- Interactive web interface built with Streamlit
- Classifies email text into Spam or Not Spam
- Uses a pre-trained model (spam_model.pkl) and vectorizer (vectorizer.pkl)
- Instant feedback with clear visual indicators
How It Works
- Vectorizer converts input email text into numerical features.
- Model predicts whether the email is spam (1) or not spam (0).
- Streamlit UI displays the result with clear visual feedback.
