# SMS Spam Detection System

A complete end-to-end Machine Learning project to classify SMS messages and emails into Spam or Not Spam (Ham).

## Features
- **Data Pipeline**: Automatically fetches the UCI SMS Spam Collection dataset, cleans, and preprocesses the text.
- **Machine Learning**: Uses TF-IDF feature extraction and a Multinomial Naive Bayes classification model.
- **Web App UI**: A visually appealing and interactive Streamlit application.

## Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn
- NLTK
- Streamlit

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the training script to download the dataset, process the text, train the model, and save the artifacts (`.pkl` files):
```bash
python train_model.py
```
*This step evaluates the model, displaying Accuracy, Precision, Recall, and F1-score.*

### 3. Run the Web Application
Launch the Streamlit interface using Python:
```bash
python -m streamlit run app.py
```

## Deployment (Streamlit Community Cloud)
This project is ready for deployment.
1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account and select this repository.
4. Set the main file path to `app.py`.
5. Click **Deploy**. The platform will automatically install packages from `requirements.txt`.
