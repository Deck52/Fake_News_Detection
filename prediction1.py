import pandas as pd
import numpy as np
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download necessary NLP resources
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Preprocessing Function
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# ✅ Load Dataset
file_path = 'G:/My Drive/6th sem/BCS602 Machine Learning/ML project/WELFake_Dataset.csv'

try:
    data = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print("❌ Error loading dataset:", e)
    sys.exit()

# Fill missing values with empty strings
data.fillna('', inplace=True)

# Combine Title & Text for Better Learning
data['combined_text'] = data['title'] + " " + data['text']
data['combined_text'] = data['combined_text'].apply(clean_text)  # Apply preprocessing

# Extract Features & Labels
X = data['combined_text']
y = data['label']

# ✅ Train-Test Split for Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ✅ Create Optimized Model Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2), min_df=5)),  # Bigram + Filtering
    ('model', LogisticRegression(max_iter=500, class_weight='balanced', C=1.0, solver='liblinear'))  # Balanced Class Weight
])

# ✅ Train the Model
pipeline.fit(X_train, y_train)

# ✅ Evaluate Performance
y_pred = pipeline.predict(X_test)
print("\n🔹 Model Performance Report:")
print(classification_report(y_test, y_pred))

# ✅ Prediction Function
def predict_news(title, text):
    if not title.strip() or not text.strip():
        print("⚠️ Error: Both title and text must be provided.")
        return
    
    input_text = clean_text(title + " " + text)
    prediction_prob = pipeline.predict_proba([input_text])[0][1]
    
    print(f"\n📊 Prediction Probability: {prediction_prob * 100:.2f}% Real")
    
    if prediction_prob > 0.5:
        print("✅ **Prediction: Real News (1)**")
    else:
        print("❌ **Prediction: Fake News (0)**")

# ✅ Get User Input
sample_title = input("\nEnter the title of the news article: ").strip()
sample_text = input("Enter the content of the news article: ").strip()

# Predict Result
predict_news(sample_title, sample_text)
