import pandas as pd
import re
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Add custom IT-related stopwords
stop_words.update(['via', 'using', 'due', 'issue', 'users', 'affecting', 'needs', 'checked', 'investigated'])

# Preprocess text
def preprocess(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'\W+', ' ', text).strip()
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 3]
    return tokens

# Extract n-grams (bigrams and trigrams)
def get_ngrams(tokens, n=2):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Main function for unsupervised analysis
def analyze_incidents_unsupervised(csv_file_path, num_topics=10, num_clusters=10):
    # Load CSV
    try:
        df = pd.read_csv(csv_file_path)
        if 'SHORT_DESCRIPTION' not in df.columns or 'SUB_CATEGORY' not in df.columns:
            raise KeyError("CSV must contain 'SHORT_DESCRIPTION' and 'SUB_CATEGORY' columns.")
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None, None
    except KeyError as e:
        print(f"Error: {e}")
        return None, None

    # Preprocess short descriptions
    df['tokens'] = df['SHORT_DESCRIPTION'].apply(preprocess)
    df['clean_text'] = df['tokens'].apply(lambda x: ' '.join(x))

    # Filter out empty documents
    df = df[df['clean_text'].str.len() > 0]

    # Topic Modeling with LDA
    dictionary = corpora.Dictionary(df['tokens'])
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens']]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)

    # Get dominant topic for each document
    df['topic'] = [max(lda_model[doc], key=lambda x: x[1])[0] if doc else -1 for doc in corpus]

    # Extract patterns from LDA
    topic_patterns = []
    for topic_id in range(num_topics):
        top_words = [word for word, _ in lda_model.show_topic(topic_id, topn=5)]
        pattern = ' '.join(top_words[:2])
        ticket_count = len(df[df['topic'] == topic_id])
        group_name = df[df['topic'] == topic_id]['SUB_CATEGORY'].mode().iloc[0] if ticket_count > 0 else 'Unknown'
        if ticket_count > 0:
            topic_patterns.append((pattern, ticket_count, group_name))

    # Clustering with K-Means on TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,3), stop_words='english')
    X = vectorizer.fit_transform(df['clean_text'])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    # Get top terms per cluster
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    cluster_patterns = []
    for cluster_id in range(num_clusters):
        cluster_docs = df[df['cluster'] == cluster_id]
        if len(cluster_docs) == 0:
            continue
        top_terms = [terms[ind] for ind in order_centroids[cluster_id, :3]]
        pattern = top_terms[0] if len(top_terms) == 1 else ' '.join(top_terms[:2])
        ticket_count = len(cluster_docs)
        group_name = cluster_docs['SUB_CATEGORY'].mode().iloc[0]
        cluster_patterns.append((pattern, ticket_count, group_name))

    # Combine patterns, remove duplicates
    all_patterns = topic_patterns + cluster_patterns
    result_df = pd.DataFrame(all_patterns, columns=['Pattern', 'Number of Tickets', 'Group Name'])
    result_df = result_df[result_df['Number of Tickets'] > 0]
    result_df = result_df.sort_values('Number of Tickets', ascending=False).drop_duplicates('Pattern')

    # Prepare chart data
    chart_data = {
        "type": "bar",
        "data": {
            "labels": result_df['Pattern'].tolist()[:10],  # Top 10 patterns
            "datasets": [{
                "label": "Number of Tickets",
                "data": result_df['Number of Tickets'].tolist()[:10],
                "backgroundColor": ["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40", "#C9CBCF", "#7BC225", "#F7C6C7", "#6F4E37"],
                "borderColor": ["#2A80B9", "#D0485B", "#D4A017", "#3A9C9C", "#7B4ACC", "#D87C30", "#A9A9B2", "#5A9A1B", "#D6A6A7", "#4A2F1E"],
                "borderWidth": 1
            }]
        },
        "options": {
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": {"display": True, "text": "Number of Tickets"}
                },
                "x": {
                    "title": {"display": True, "text": "Pattern"}
                }
            },
            "plugins": {
                "legend": {"display": False},
                "title": {"display": True, "text": "Top IT Incident Patterns"}
            }
        }
    }

    return result_df, chart_data

# Example usage
if __name__ == "__main__":
    csv_file_path = 'incident_data.csv'  # Replace with your CSV path
    results, chart = analyze_incidents_unsupervised(csv_file_path, num_topics=10, num_clusters=10)
    if results is not None:
        print(results.to_markdown(index=False))
        print("\nChart of top patterns:")
        print("```chartjs")
        print(chart)
        print("```")
