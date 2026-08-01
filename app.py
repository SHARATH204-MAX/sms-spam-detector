import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure stopwords are downloaded (especially needed in deployment)
nltk.download('stopwords', quiet=True)

# Load the trained model and vectorizer
try:
    model = joblib.load('spam_classifier_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please train the model first by running `python train_model.py`.")
    st.stop()

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit App UI
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

st.title("📩 SMS Spam Detection System")
st.markdown("This machine learning app classifies SMS messages or emails into **Spam** or **Not Spam (Ham)**.")

st.markdown("### Enter your message below:")
user_input = st.text_area("Type the message here:", height=150)

if st.button("Predict", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a valid message!")
    else:
        with st.spinner('Analyzing...'):
            # Preprocess the input
            transformed_sms = preprocess_text(user_input)
            
            # Vectorize
            vector_input = tfidf.transform([transformed_sms])
            
            # Predict and get probabilities
            result = model.predict(vector_input)[0]
            probabilities = model.predict_proba(vector_input)[0]
            spam_prob = probabilities[1] * 100
            ham_prob = probabilities[0] * 100
            
            # Display Result
            st.markdown("---")
            if result == 1:
                st.error("🚨 **Prediction: SPAM**")
            else:
                st.success("✅ **Prediction: NOT SPAM (Ham)**")
                
            # Show Confidence
            st.markdown("### Model Confidence")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Spam Probability: **{spam_prob:.2f}%**")
                st.progress(int(spam_prob))
            with col2:
                st.info(f"Ham Probability: **{ham_prob:.2f}%**")
                st.progress(int(ham_prob))

            # Explainability: Why did the model make this decision?
            # We look at the words in the input and their TF-IDF scores
            st.markdown("### Text Analysis Breakdown")
            words = transformed_sms.split()
            if len(words) > 0:
                # Find the words from the input that the model considers most "Spammy"
                # For Naive Bayes, we can look at the feature log probabilities
                feature_names = tfidf.get_feature_names_out()
                
                word_importance = {}
                for word in words:
                    if word in feature_names:
                        idx = tfidf.vocabulary_[word]
                        # Log probability of the word given Spam (class 1) vs Ham (class 0)
                        spam_score = model.feature_log_prob_[1][idx]
                        ham_score = model.feature_log_prob_[0][idx]
                        # Importance is how much more likely it is in Spam than Ham
                        word_importance[word] = spam_score - ham_score
                
                if word_importance:
                    # Sort by spam importance
                    sorted_words = sorted(word_importance.items(), key=lambda item: item[1], reverse=True)
                    top_spam_words = [w[0] for w in sorted_words if w[1] > 0][:5]
                    
                    if top_spam_words and result == 1:
                        st.warning(f"**Keywords flagged as highly suspicious:** {', '.join(top_spam_words)}")
                    elif result == 0:
                        st.success("No highly suspicious keywords detected.")
                else:
                    st.write("No significant keywords found in the vocabulary.")
