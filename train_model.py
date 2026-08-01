import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import urllib.request
import zipfile
import os

# Download NLTK data
nltk.download('stopwords')

def download_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = "smsspamcollection.zip"
    data_path = "SMSSpamCollection"
    
    if not os.path.exists(data_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)
    return data_path

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Tokenize (simple split)
    words = text.split()
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def main():
    data_path = download_data()
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])
    
    # Data Cleaning & Duplicate Removal
    print("Initial shape:", df.shape)
    df.drop_duplicates(inplace=True)
    print("Shape after dropping duplicates:", df.shape)
    
    # Map labels to binary
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Text Preprocessing
    print("Preprocessing text...")
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    # Feature Extraction
    print("Extracting features with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['processed_message']).toarray()
    y = df['label'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("------------------------\n")
    
    # Save Model and Vectorizer
    print("Saving model and vectorizer...")
    joblib.dump(model, 'spam_classifier_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print("Saved to 'spam_classifier_model.pkl' and 'tfidf_vectorizer.pkl'.")

if __name__ == "__main__":
    main()
