import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# Load the dataset
file_path = 'G:/My Drive/6th sem/BCS602 Machine Learning/ML project/WELFake_Dataset.csv'

try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# Check for missing values and drop rows with missing values
data.dropna(inplace=True)

# Using 'title' and 'text' columns for analysis
X_title = data['title']
X_text = data['text']
y = data['label']

# Vectorizing title and text separately using TF-IDF
vectorizer_title = TfidfVectorizer(stop_words='english', max_df=0.7)
X_title = vectorizer_title.fit_transform(X_title)

vectorizer_text = TfidfVectorizer(stop_words='english', max_df=0.7)
X_text = vectorizer_text.fit_transform(X_text)

# Train Logistic Regression model for title and text separately
model_title = LogisticRegression(max_iter=200)
model_title.fit(X_title, y)

model_text = LogisticRegression(max_iter=200)
model_text.fit(X_text, y)

# Prediction function with separate analysis for title and text
def predict_news_with_separate_analysis(title, text):
    # Vectorize the title and text provided by the user
    title_vector = vectorizer_title.transform([title])
    text_vector = vectorizer_text.transform([text])
    
    # Get probabilities for both title and text
    title_prob = model_title.predict_proba(title_vector)[0]
    text_prob = model_text.predict_proba(text_vector)[0]
    
    # Combine probabilities: for example, take the average
    combined_prob = np.mean([title_prob[1], text_prob[1]])  # Average of 'real' probabilities
    
    print(f"Title Prediction Probabilities (Fake = 0, Real = 1): {title_prob[1] * 100:.2f}% (Real)")
    print(f"Text Prediction Probabilities (Fake = 0, Real = 1): {text_prob[1] * 100:.2f}% (Real)")
    print(f"Combined Prediction Probability (Fake = 0, Real = 1): {combined_prob * 100:.2f}% (Real)")
    
    # Return final prediction (consider the combined probabilities)
    if combined_prob > 0.5:
        print("Prediction: Real News (1)")
    else:
        print("Prediction: Fake News (0)")

# User input for title and text
sample_title = input("Enter the title of the news article: ")
sample_text = input("Enter the content of the news article: ")

# Make prediction for the input title and text
predict_news_with_separate_analysis(sample_title, sample_text)