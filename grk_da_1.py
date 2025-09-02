```python
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
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
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()
    text = re.sub(r'\W+', ' ', text).strip()
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 3]
    return tokens

# Main function for unsupervised analysis
def analyze_incidents_unsupervised(csv_file_path, output_csv_path='incident_patterns.csv', max_clusters_per_group=50):
    # Load CSV with only required columns for memory efficiency
    try:
        df = pd.read_csv(csv_file_path, usecols=['INCIDENT_NUMBER', 'SHORT_DESCRIPTION', 'SUB_CATEGORY', 'ASSIGNMENT_GROUP'], low_memory=False)
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None, None
    except ValueError as e:
        print(f"Error: Missing required columns - {e}")
        return None, None

    # Handle null/empty values
    df['INCIDENT_NUMBER'] = df['INCIDENT_NUMBER'].fillna('Unknown').astype(str)
    df['SUB_CATEGORY'] = df['SUB_CATEGORY'].fillna('Unknown')
    df['ASSIGNMENT_GROUP'] = df['ASSIGNMENT_GROUP'].fillna('Unknown')
    df['SHORT_DESCRIPTION'] = df['SHORT_DESCRIPTION'].fillna('')

    # Filter rows with valid SHORT_DESCRIPTION for analysis
    df_analysis = df[df['SHORT_DESCRIPTION'].str.strip() != ''].copy()

    if df_analysis.empty:
        print("Error: No valid SHORT_DESCRIPTION entries for analysis.")
        # Output incidents with no description
        missing_docs = df[df['SHORT_DESCRIPTION'].str.strip() == ''].copy()
        if not missing_docs.empty:
            missing_results = pd.DataFrame({
                'incident number': missing_docs['INCIDENT_NUMBER'],
                'pattern': 'No Description',
                'group name': missing_docs['SUB_CATEGORY'],
                'assignment group': missing_docs['ASSIGNMENT_GROUP']
            })
            missing_results.to_csv(output_csv_path, index=False)
            print(f"Results saved to {output_csv_path} (all incidents have no description)")
        return None, None

    # Preprocess short descriptions
    df_analysis['tokens'] = df_analysis['SHORT_DESCRIPTION'].apply(preprocess)
    df_analysis['clean_text'] = df_analysis['tokens'].apply(lambda x: ' '.join(x))

    # Filter out empty documents after preprocessing
    df_analysis = df_analysis[df_analysis['clean_text'].str.len() > 0]

    if df_analysis.empty:
        print("Error: No valid preprocessed text for analysis.")
        # Output incidents with no description
        missing_docs = df[df['SHORT_DESCRIPTION'].str.strip() == ''].copy()
        if not missing_docs.empty:
            missing_results = pd.DataFrame({
                'incident number': missing_docs['INCIDENT_NUMBER'],
                'pattern': 'No Description',
                'group name': missing_docs['SUB_CATEGORY'],
                'assignment group': missing_docs['ASSIGNMENT_GROUP']
            })
            missing_results.to_csv(output_csv_path, index=False)
            print(f"Results saved to {output_csv_path} (all incidents have no valid text)")
        return None, None

    # Group by ASSIGNMENT_GROUP and perform clustering per group
    results = []
    for assignment_group, group in df_analysis.groupby('ASSIGNMENT_GROUP'):
        if len(group) < 2:
            # Too small to cluster
            for _, row in group.iterrows():
                results.append({
                    'incident number': row['INCIDENT_NUMBER'],
                    'pattern': 'Isolated Incident',
                    'group name': row['SUB_CATEGORY'],
                    'assignment group': assignment_group
                })
            continue

        # Vectorize text for the group
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,3), stop_words='english')
        X = vectorizer.fit_transform(group['clean_text'])

        # Determine number of clusters: up to max_clusters_per_group
        num_clusters = min(max_clusters_per_group, max(1, len(group) // 2)) if len(group) < 50 else max_clusters_per_group

        # Clustering with MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=1000)
        group['cluster'] = kmeans.fit_predict(X)

        # Get top terms per cluster
        terms = vectorizer.get_feature_names_out()
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        for cluster_id in range(num_clusters):
            cluster_docs = group[group['cluster'] == cluster_id]
            if len(cluster_docs) == 0:
                continue
            top_terms = [terms[ind] for ind in order_centroids[cluster_id, :4]]  # Top 4 terms for more context
            pattern = ' '.join(top_terms)
            for _, row in cluster_docs.iterrows():
                results.append({
                    'incident number': row['INCIDENT_NUMBER'],
                    'pattern': pattern,
                    'group name': row['SUB_CATEGORY'],
                    'assignment group': assignment_group
                })

    # Combine results into DataFrame
    result_df = pd.DataFrame(results)

    # Include incidents with null/empty SHORT_DESCRIPTION in output
    missing_docs = df[df['SHORT_DESCRIPTION'].str.strip() == ''].copy()
    if not missing_docs.empty:
        missing_results = pd.DataFrame({
            'incident number': missing_docs['INCIDENT_NUMBER'],
            'pattern': 'No Description',
            'group name': missing_docs['SUB_CATEGORY'],
            'assignment group': missing_docs['ASSIGNMENT_GROUP']
        })
        result_df = pd.concat([result_df, missing_results], ignore_index=True)

    # Save to CSV
    result_df.to_csv(output_csv_path, index=False, columns=['incident number', 'pattern', 'group name', 'assignment group'])
    print(f"Results saved to {output_csv_path}")

    # Prepare chart data (for pattern counts, excluding 'No Description' and 'Isolated Incident')
    pattern_counts = result_df[~result_df['pattern'].isin(['No Description', 'Isolated Incident'])].groupby('pattern').size().reset_index(name='Number of Tickets')
    pattern_counts = pattern_counts.sort_values('Number of Tickets', ascending=False)
    if pattern_counts.empty:
        print("Warning: No patterns to display in chart.")
        chart_data = None
    else:
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
        print(results.head(10).to_markdown(index=False))  # Show first 10 rows
        if chart:
            print("\nChart of top patterns:")
            print("```chartjs")
            print(chart)
            print("```")
```