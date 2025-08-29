```python
import pandas as pd
import re
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Preprocess text (efficient regex, handle non-strings)
def preprocess(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'\W+', ' ', text).strip()
    return text.split()

# Main function for unsupervised analysis
def analyze_incidents_unsupervised(csv_file_path, num_topics=5, num_clusters=5):
    # Load CSV
    try:
        df = pd.read_csv(csv_file_path)
        if 'description' not in df.columns:
            raise KeyError("CSV must contain a 'description' column.")
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None
    except KeyError as e:
        print(f"Error: {e}")
        return None

    # Preprocess descriptions
    df['tokens'] = df['description'].apply(preprocess)
    df['clean_text'] = df['tokens'].apply(lambda x: ' '.join(x))

    # Topic Modeling with LDA
    dictionary = corpora.Dictionary(df['tokens'])
    corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens']]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)

    # Get dominant topic for each document
    df['topic'] = [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus]

    # Extract top words per topic as patterns
    topic_patterns = []
    for topic_id in range(num_topics):
        top_words = [word for word, _ in lda_model.show_topic(topic_id, topn=3)]
        pattern = ' '.join(top_words[:2])  # Use top 2 words as pattern
        ticket_count = len(df[df['topic'] == topic_id])
        group_name = top_words[0].capitalize()  # Use top word as group name
        topic_patterns.append((pattern, ticket_count, group_name))

    # Clustering with K-Means on TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['clean_text'])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    # Get top terms per cluster
    terms = vectorizer.get_feature_names_out()
    cluster_patterns = []
    for cluster_id in range(num_clusters):
        cluster_docs = df[df['cluster'] == cluster_id]
        all_words = [word for tokens in cluster_docs['tokens'] for word in tokens]
        word_freq = Counter(all_words)
        top_words = [word for word, _ in word_freq.most_common(3)]
        pattern = ' '.join(top_words[:2])
        ticket_count = len(cluster_docs)
        group_name = top_words[0].capitalize()
        cluster_patterns.append((pattern, ticket_count, group_name))

    # Combine patterns, remove duplicates
    all_patterns = topic_patterns + cluster_patterns
    result_df = pd.DataFrame(all_patterns, columns=['Pattern', 'Number of Tickets', 'Group Name'])
    result_df = result_df.sort_values('Number of Tickets', ascending=False).drop_duplicates('Pattern')

    return result_df

# Example usage
if __name__ == "__main__":
    csv_file_path = 'your_file.csv'  # Replace with your CSV path
    results = analyze_incidents_unsupervised(csv_file_path, num_topics=5, num_clusters=5)
    if results is not None:
        print(results.to_markdown(index=False))
```