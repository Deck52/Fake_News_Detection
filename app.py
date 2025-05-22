# import streamlit as st
# import joblib
# import re
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer, SnowballStemmer

# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load model and vectorizer
# model = joblib.load("genre_classifier.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# # Text cleaning
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
# stemmer = SnowballStemmer('english')

# def clean(text):
#     text = text.lower()
#     text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
#     text_tokens = word_tokenize(text)
#     text = " ".join([word for word in text_tokens if word not in stop_words and len(word) > 3])
#     text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
#     text = re.sub("<.*?>", " ", text)
#     text = re.sub("\n", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     return text

# def preprocess(text):
#     tokens = word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     tokens = [stemmer.stem(word) for word in tokens]
#     return " ".join(tokens)

# # Streamlit app UI
# st.title("📚 Book Genre Classifier")
# st.markdown("Enter a book title or summary to predict its genre.")

# user_input = st.text_area("Enter book title or description here:")

# if st.button("Predict Genre"):
#     if user_input.strip() == "":
#         st.warning("Please enter valid text.")
#     else:
#         cleaned = preprocess(clean(user_input))
#         vector = vectorizer.transform([cleaned])
        
#         predicted_label = model.predict(vectorized_input)
#         predicted_genre = le.inverse_transform(predicted_label)
#         print("Predicted Genre:", predicted_genre[0])
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import streamlit as st

# Load the dataset
file_path = 'WELFake_Dataset.csv'

try:
    data = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Fill missing values
data.fillna('', inplace=True)

# Extract features and labels
X_title = data['title']
X_text = data['text']
y = data['label']

# Define a reusable pipeline
def create_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('model', LogisticRegression(max_iter=300, class_weight='balanced'))
    ])

# Train models
model_title = create_pipeline()
model_title.fit(X_title, y)

model_text = create_pipeline()
model_text.fit(X_text, y)

# Streamlit app UI
st.title("📰 Fake News Detector")

st.markdown("""
Enter the **title** and **content** of a news article below to predict whether it's likely **Real** or **Fake**.
""")

title_input = st.text_input("📝 News Title")
text_input = st.text_area("📄 News Content")

if st.button("🔍 Predict"):
    if not title_input.strip() or not text_input.strip():
        st.warning("Please enter both title and content.")
    else:
        # Get prediction probabilities
        title_prob = model_title.predict_proba([title_input])[0][1]
        text_prob = model_text.predict_proba([text_input])[0][1]
        combined_prob = 0.6 * text_prob + 0.4 * title_prob

        st.markdown(f"**Title Prediction:** `{title_prob * 100:.2f}%` Real")
        st.markdown(f"**Text Prediction:** `{text_prob * 100:.2f}%` Real")
        st.markdown(f"**Combined Probability:** `{combined_prob * 100:.2f}%` Real")

        if combined_prob > 0.5:
            st.success("✅ **Prediction: Real News (1)**")
        else:
            st.error("❌ **Prediction: Fake News (0)**")

