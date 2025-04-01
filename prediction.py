import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys

# Load the dataset
file_path = 'G:/My Drive/6th sem/BCS602 Machine Learning/ML project/WELFake_Dataset.csv'

try:
    data = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print("❌ Error loading dataset:", e)
    sys.exit()

# Fill missing values with empty strings to avoid data loss
data.fillna('', inplace=True)

# Extract features and labels
X_title = data['title']
X_text = data['text']
y = data['label']

# Define TF-IDF Vectorizer and Model Pipeline
def create_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('model', LogisticRegression(max_iter=300, class_weight='balanced'))
    ])

# Train models for title and text
model_title = create_pipeline()
model_title.fit(X_title, y)

model_text = create_pipeline()
model_text.fit(X_text, y)

# Function for predicting news authenticity
def predict_news(title, text):
    if not title.strip() or not text.strip():
        print("⚠️ Error: Both title and text must be provided.")
        return
    
    # Vectorize input
    title_prob = model_title.predict_proba([title])[0][1]
    text_prob = model_text.predict_proba([text])[0][1]
    
    # Logarithmic weighted combination of probabilities
    combined_prob = (0.6 * text_prob) + (0.4 * title_prob)
    
    # Print prediction results
    print(f"\n🔍 Title Prediction: {title_prob * 100:.2f}% Real")
    print(f"📰 Text Prediction: {text_prob * 100:.2f}% Real")
    print(f"📊 Combined Probability: {combined_prob * 100:.2f}% Real")
    
    if combined_prob > 0.5:
        print("✅ **Prediction: Real News (1)**")
    else:
        print("❌ **Prediction: Fake News (0)**")

# Get user input
sample_title = input("\nEnter the title of the news article: ").strip()
sample_text = input("\nEnter the content of the news article: ").strip()

# Predict result
predict_news(sample_title, sample_text)
