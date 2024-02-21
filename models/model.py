
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import jaccard_score

def load_model():
    df = pd.read_csv('C:\\Users\\diksh\\output.csv')
    documents = df['value'].tolist()
    
    def preprocess_text(value):
        # Remove punctuation and special characters
        value = re.sub(r'[^\w\s]', '', value)
        # Convert to lowercase
        value = value.lower()
        return value

    preprocessed_documents = [preprocess_text(doc) for doc in documents]
    tokenized_documents = [doc.split() for doc in preprocessed_documents]
    tokenized_documents = [" ".join(doc) for doc in tokenized_documents]
    
    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_documents)
    
    nmf = NMF(n_components=10, random_state=1)
    nmf.fit(tfidf_matrix)
    
    # Get the top words for each topic
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_words_per_topic = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_words_idx = topic.argsort()[-10:][::-1]  # Get the indices of the top 10 words
        top_words = [feature_names[i] for i in top_words_idx]
        top_words_per_topic.append(top_words)
    
    # Post-processing: Remove similar terms
    
    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1) + len(set2) - intersection
        return intersection / union
    
    def remove_similar_terms(topic_words, threshold=0.7):
        unique_words = []
        for words in topic_words:
            if not any(jaccard_similarity(set(words), set(unique)) >= threshold for unique in unique_words):
                unique_words.append(words)
        return unique_words
    
    # Apply the remove_similar_terms function to the top words for each topic
    filtered_topics = [remove_similar_terms(words) for words in top_words_per_topic]

    # Return the variables you want to use in other parts of your code
    return filtered_topics, tfidf_vectorizer, nmf


    