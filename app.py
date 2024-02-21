# app.py

from flask import Flask, render_template, request, jsonify
from models.model import load_model
from models import model
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import jaccard_score

app = Flask(__name__)

# Load the model and get the variables
filtered_topics, tfidf_vectorizer, nmf = load_model()

# Create a function to predict topics
def predict_topic(text):
    
        def preprocess_text(value):
            value = re.sub(r'[^\w\s]', '', value)
            value = value.lower()
            return value

        preprocessed_text = preprocess_text(text)
        tokenized_text = " ".join(preprocessed_text.split())

        # Use the model to predict the topic
        new_data = [tokenized_text]
        new_tfidf_matrix = tfidf_vectorizer.transform(new_data)
        new_topic_distribution = nmf.transform(new_tfidf_matrix)
        max_topic_index = new_topic_distribution.argmax()
        max_topic_keywords = filtered_topics[max_topic_index]
        return max_topic_keywords[0]

@app.route("/")
def index():
    return render_template("index.html")       
    
@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    # Call the predict_topic function with the input text
    max_topic_keywords = predict_topic(text)

    # Return the result as JSON
    return jsonify({'max_topic_keywords': max_topic_keywords})

if __name__ == '__main__':
    app.run(debug=True , port=5000)
