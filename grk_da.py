import pandas as pd
import re
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

# Main function for unsupervised analysis
def analyze_incidents_unsupervised(csv_file_path, output_csv_path='incident_patterns.csv', num_topics=10, num_clusters=10):
    # Load CSV
    try:
        df = pd.read_csv(csv_file_path)
        required_columns = ['INCIDENT_NUMBER', 'SHORT_DESCRIPTION', 'SUB_CATEGORY', 'ASSIGNMENT_GROUP']
        if not all(col in df.columns for col in required_columns):
            raise KeyError("CSV must contain 'INCIDENT_NUMBER', 'SHORT_DESCRIPTION', 'SUB_CATEGORY', and 'ASSIGNMENT_GROUP' columns.")
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
    topic_results = []
    for topic_id in range(num_topics):
        top_words = [word for word, _ in lda_model.show_topic(topic_id, topn=5)]
        pattern = ' '.join(top_words[:2])
        topic_docs = df[df['topic'] == topic_id]
        if len(topic_docs) > 0:
            for _, row in topic_docs.iterrows():
                topic_results.append({
                    'incident number': row['INCIDENT_NUMBER'],
                    'pattern': pattern,
                    'group name': row['SUB_CATEGORY'],
                    'assignment group': row['ASSIGNMENT_GROUP']
                })

    # Clustering with K-Means on TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,3), stop_words='english')
    X = vectorizer.fit_transform(df['clean_text'])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    # Get top terms per cluster
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    cluster_results = []
    for cluster_id in range(num_clusters):
        cluster_docs = df[df['cluster'] == cluster_id]
        if len(cluster_docs) == 0:
            continue
        top_terms = [terms[ind] for ind in order_centroids[cluster_id, :3]]
        pattern = top_terms[0] if len(top_terms) == 1 else ' '.join(top_terms[:2])
        for _, row in cluster_docs.iterrows():
            cluster_results.append({
                'incident number': row['INCIDENT_NUMBER'],
                'pattern': pattern,
                'group name': row['SUB_CATEGORY'],
                'assignment group': row['ASSIGNMENT_GROUP']
            })

    # Combine results
    all_results = topic_results + cluster_results
    result_df = pd.DataFrame(all_results)
    result_df = result_df.drop_duplicates(subset=['incident number', 'pattern'])

    # Save to CSV
    result_df.to_csv(output_csv_path, index=False, columns=['incident number', 'pattern', 'group name', 'assignment group'])
    print(f"Results saved to {output_csv_path}")

    # Prepare chart data (for pattern counts)
    pattern_counts = result_df.groupby('pattern').size().reset_index(name='Number of Tickets')
    pattern_counts = pattern_counts.sort_values('Number of Tickets', ascending=False)
    chart_data = {
        "type": "bar",
        "data": {
            "labels": pattern_counts['pattern'].tolist()[:10],
            "datasets": [{
                "label": "Number of Tickets",
                "data": pattern_counts['Number of Tickets'].tolist()[:10],
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
    csv_file_path = 'incidents.csv'  # Replace with your input CSV path
    output_csv_path = 'incident_patterns.csv'  # Output CSV file
    results, chart = analyze_incidents_unsupervised(csv_file_path, output_csv_path)
    if results is not None:
        print(results.to_markdown(index=False))
        print("\nChart of top patterns:")
        print("```chartjs")
        print(chart)
        print("```")
