import pandas as pd
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(filtered_tokens)

# Lemmatization function
def lemmatize_text(text):
    return ' '.join([WordNetLemmatizer().lemmatize(word) for word in text.split()])

# Load the model and vectorizer
model = joblib.load('best_log_model.pkl')

# Function to predict on new data
def predict_new_text(text):
    processed_text = preprocess_text(text)
    lemmatized_text = lemmatize_text(processed_text)
    vectorized_text = tfidf.transform([lemmatized_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]



if __name__ == "__main__":
    new_text = "This is a great movie with excellent acting!"
    prediction = predict_new_text(new_text)
    print(f"The prediction for the input text is: {prediction}")

