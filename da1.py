import pandas as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from langdetect import detect_langs
import io
import csv
import warnings
warnings.filterwarnings('ignore')

# Function to safely detect language
def detect_language(text):
    try:
        if pd.isna(text) or str(text).strip() == '':
            return 'unknown'
        langs = detect_langs(str(text))
        if langs:
            lang = langs[0].lang
            prob = langs[0].prob
            if prob > 0.9:  # Threshold for confidence
                return lang if lang != 'en' else 'en'
        return 'unknown'
    except Exception:
        return 'unknown'

# Function to preprocess text, handling multiline
def preprocess_text(text):
    if pd.isna(text):
        return ''
    # Replace line breaks with spaces and clean
    text = str(text).replace('\n', ' ').replace('\r', ' ').lower().strip()
    return text

# Function to generate professional cluster names
def generate_cluster_name(keywords):
    if not keywords or len(keywords) == 0:
        return "Unclassified Issue"
    # Use first two keywords and format for professionalism
    if len(keywords) >= 2:
        return f"{keywords[0].replace(',', '').title()} and {keywords[1].replace(',', '').title()} Issues"
    return f"{keywords[0].replace(',', '').title()} Issues"

# Function to attempt reading CSV with multiple encodings
def read_csv_with_encodings(uploaded_file):
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'windows-1252']
    for encoding in encodings:
        try:
            uploaded_file.seek(0)  # Reset file pointer
            df = pd.read_csv(uploaded_file, quoting=csv.QUOTE_ALL, escapechar='\\', encoding=encoding)
            st.info(f"Successfully read CSV with encoding: {encoding}")
            return df, encoding
        except UnicodeDecodeError as e:
            st.warning(f"Failed to read CSV with {encoding}: {str(e)}")
        except pd.errors.ParserError as e:
            st.warning(f"Parsing error with {encoding}: {str(e)}. Ensure fields with line breaks are quoted.")
        except Exception as e:
            st.warning(f"Unexpected error with {encoding}: {str(e)}")
    st.error(f"Failed to read CSV with encodings: {', '.join(encodings)}. Check file encoding or quoting of multiline fields.")
    return None, None

# Main analysis function
@st.cache_data
def analyze_data(uploaded_file, max_features, n_clusters_base, ngram_max, pattern_field, stat_fields):
    try:
        # Read CSV with encoding detection
        df, used_encoding = read_csv_with_encodings(uploaded_file)
        if df is None:
            return None, None, None, None
        
        # Normalize column names to lowercase for case insensitivity
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Required columns (handle possible variations)
        col_mapping = {
            'number': ['number', 'incident_number', 'incident id'],
            'description': ['description', 'desc', 'long_description'],
            'short_description': ['short_description', 'short desc', 'summary'],
            'category': ['category'],
            'sub_category': ['sub_category', 'subcategory'],
            'assignment_group': ['assignment_group', 'group', 'assigned_to_group']
        }
        
        # Find actual column names
        actual_cols = {}
        for standard, variations in col_mapping.items():
            for var in variations:
                if var in df.columns:
                    actual_cols[standard] = var
                    break
            if standard not in actual_cols:
                st.error(f"Required column '{standard}' (or variations {variations}) not found. Available: {list(df.columns)}")
                return None, None, None, None
        
        # Rename to standard
        df = df.rename(columns=actual_cols)
        
        # Validate pattern field
        if pattern_field not in df.columns:
            st.error(f"Selected pattern field '{pattern_field}' not found in CSV.")
            return None, None, None, None
        
        # Handle large dataset: warn if >1M
        if len(df) > 1000000:
            st.warning(f"Dataset has {len(df)} rows. Processing all, but clustering may take time.")
        
        # Detect languages and filter English
        st.info("Detecting languages...")
        df['detected_lang'] = df[pattern_field].apply(lambda x: detect_language(x))
        english_mask = df['detected_lang'] == 'en'
        analyzed_df = df[english_mask].copy()
        omitted_count = len(df) - len(analyzed_df)
        analyzed_count = len(analyzed_df)
        
        if analyzed_count == 0:
            st.error("No English records found.")
            return None, None, None, None
        
        # Preprocess text for pattern analysis
        analyzed_df['analysis_text'] = analyzed_df[pattern_field].apply(preprocess_text)
        
        # Group by assignment_group
        groups = analyzed_df.groupby('assignment_group')
        
        # Prepare output DF
        output_cols = ['number', 'description', 'short_description', 'category', 'sub_category', 'assignment_group']
        output_df = analyzed_df[output_cols].copy()
        output_df['pattern_keywords'] = np.nan  # Placeholder for keywords
        
        # For each group, perform clustering
        st.info("Performing clustering per assignment group...")
        cluster_results = {}
        for group_name, group_data in groups:
            if len(group_data) < 2:  # Skip small groups
                continue
            
            texts = group_data['analysis_text'].fillna('').tolist()
            
            # TF-IDF Vectorization with configurable parameters
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, ngram_max))
            try:
                X = vectorizer.fit_transform(texts)
                
                # Clustering with MiniBatchKMeans
                n_clusters = min(n_clusters_base, len(group_data) // 10)  # Adaptive clusters
                if n_clusters < 2:
                    n_clusters = 2
                
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
                clusters = kmeans.fit_predict(X)
                
                # Extract top keywords and incident counts per cluster
                patterns = {}
                feature_names = vectorizer.get_feature_names_out()
                for i in range(n_clusters):
                    cluster_mask = clusters == i
                    cluster_size = np.sum(cluster_mask)
                    cluster_terms = X[cluster_mask].mean(axis=0).A1
                    top_indices = np.argsort(cluster_terms)[-5:][::-1]  # Top 5 terms
                    top_keywords = [feature_names[idx] for idx in top_indices]
                    cluster_name = generate_cluster_name(top_keywords)
                    patterns[cluster_name] = {'keywords': top_keywords, 'incident_count': cluster_size}
                
                # Assign keywords to output DF
                group_indices = group_data.index
                for idx, cluster in zip(group_indices, clusters):
                    cluster_name = list(patterns.keys())[cluster]
                    output_df.at[idx, 'pattern_keywords'] = ', '.join(patterns[cluster_name]['keywords'])
                
                cluster_results[group_name] = patterns
                
            except Exception as e:
                st.warning(f"Clustering failed for group {group_name}: {str(e)}")
                output_df.loc[group_data.index, 'pattern_keywords'] = 'clustering_failed'
        
        # High-level stats
        stats = {
            'total_records': len(df),
            'analyzed_records': analyzed_count,
            'omitted_records': omitted_count,
            'non_english_languages': df['detected_lang'].value_counts().to_dict(),
            'top_assignment_groups': analyzed_df['assignment_group'].value_counts().head(10).to_dict(),
            'patterns_per_group': cluster_results
        }
        
        # Prepare CSV buffer
        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        return stats, csv_data, output_df, list(analyzed_df.columns)
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}. Available columns: {list(df.columns) if 'df' in locals() else 'Not loaded'}")
        return None, None, None, None

# Function to create Excel summary
def create_excel_summary(stats, stat_fields, output_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Dataset Overview sheet
        overview_data = {
            'Metric': ['Total Records', 'Analyzed (English)', 'Omitted (Non-English/Unknown)'],
            'Value': [stats['total_records'], stats['analyzed_records'], stats['omitted_records']]
        }
        pd.DataFrame(overview_data).to_excel(writer, sheet_name='Overview', index=False)
        
        # Non-English Languages sheet
        if stats['non_english_languages']:
            non_eng_df = pd.DataFrame(list(stats['non_english_languages'].items()), columns=['Language', 'Count'])
            non_eng_df.to_excel(writer, sheet_name='Non-English Languages', index=False)
        
        # Top Assignment Groups sheet
        top_groups_df = pd.DataFrame(list(stats['top_assignment_groups'].items()), columns=['Assignment Group', 'Count'])
        top_groups_df.to_excel(writer, sheet_name='Top Assignment Groups', index=False)
        
        # Column Statistics sheets
        for col in stat_fields:
            if output_df is not None and col in output_df.columns:
                col_counts = output_df[col].value_counts().head(10).to_dict()
                col_df = pd.DataFrame(list(col_counts.items()), columns=[col.title(), 'Count'])
                col_df.to_excel(writer, sheet_name=f'{col.title()} Stats', index=False)
        
        # Patterns per Group sheets
        for group, patterns in stats['patterns_per_group'].items():
            sorted_patterns = dict(sorted(patterns.items(), key=lambda x: x[1]['incident_count'], reverse=True))
            pattern_data = [
                {'Issue Type': cluster_name, 'Top Keywords': ', '.join(pattern['keywords']), 'Incident Count': pattern['incident_count']}
                for cluster_name, pattern in sorted_patterns.items()
            ]
            pattern_df = pd.DataFrame(pattern_data)
            safe_group_name = group.replace('/', '_').replace('\\', '_')[:31]  # Excel sheet name limit
            pattern_df.to_excel(writer, sheet_name=f'{safe_group_name} Patterns', index=False)
    
    return output.getvalue()

# Streamlit UI
st.title("ServiceNow Incident Analysis Tool")

# Step 1: File upload and review
st.header("Step 1: Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
review_button = st.button("Review")

# Initialize session state
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'available_columns' not in st.session_state:
    st.session_state.available_columns = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'output_df' not in st.session_state:
    st.session_state.output_df = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'stat_fields' not in st.session_state:
    st.session_state.stat_fields = []

# Step 2: Column selection
if uploaded_file is not None and review_button:
    try:
        # Read CSV with encoding detection
        df, _ = read_csv_with_encodings(uploaded_file)
        if df is not None:
            df.columns = [col.lower().strip() for col in df.columns]
            st.session_state.df = df
            st.session_state.available_columns = list(df.columns)
    except Exception as e:
        st.error(f"Failed to read CSV: {str(e)}. Ensure fields with line breaks are quoted and file encoding is compatible (e.g., UTF-8, Latin-1).")

if st.session_state.available_columns:
    st.header("Step 2: Configure Analysis")
    
    # Clustering parameters with help
    st.sidebar.header("Clustering Parameters")
    max_features = st.sidebar.slider(
        "Max TF-IDF Features", 500, 5000, 1000, step=500,
        help="Controls the vocabulary size for text features. Higher values capture more details but increase computation time."
    )
    n_clusters_base = st.sidebar.slider(
        "Base Number of Clusters per Group", 2, 20, 5,
        help="Maximum clusters per assignment group. Higher values create more specific clusters but may overfit small groups."
    )
    ngram_max = st.sidebar.selectbox(
        "Max N-gram Range", [1, 2, 3], index=1,
        help="Maximum length of word phrases (n-grams). 1: single words; 2: up to bigrams (e.g., 'login failure'); 3: up to trigrams for more context."
    )
    
    # Column selection for pattern analysis and stats
    st.subheader("Select Fields")
    pattern_field = st.selectbox(
        "Field for Pattern Analysis",
        options=[col for col in st.session_state.available_columns if col in ['description', 'short_description']],
        index=0 if 'short_description' in st.session_state.available_columns else 1
    )
    stat_fields = st.multiselect(
        "Fields for Statistics",
        options=[col for col in st.session_state.available_columns if col not in ['number', 'description', 'short_description']],
        default=[col for col in ['category', 'sub_category', 'assignment_group'] if col in st.session_state.available_columns]
    )
    st.session_state.stat_fields = stat_fields
    
    # Analysis button
    analyze_button = st.button("Start Analysis")
    
    if analyze_button and pattern_field and stat_fields:
        with st.spinner("Analyzing data..."):
            stats, csv_data, output_df, available_columns = analyze_data(
                uploaded_file, max_features, n_clusters_base, ngram_max, pattern_field, stat_fields
            )
            st.session_state.stats = stats
            st.session_state.output_df = output_df
            st.session_state.csv_data = csv_data
            st.session_state.available_columns = available_columns if available_columns else st.session_state.available_columns

# Display results
if st.session_state.stats:
    stats = st.session_state.stats
    output_df = st.session_state.output_df
    csv_data = st.session_state.csv_data
    stat_fields = st.session_state.stat_fields
    
    # Dataset Overview (full width)
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Total Records: {stats['total_records']}")
        st.write(f"Analyzed (English): {stats['analyzed_records']}")
        st.write(f"Omitted (Non-English/Unknown): {stats['omitted_records']}")
        
        if stats['non_english_languages']:
            st.write("Top Non-English Languages:")
            non_eng_df = pd.DataFrame(list(stats['non_english_languages'].items()), 
                                      columns=['Language', 'Count']).head(5)
            st.table(non_eng_df)
    
    with col2:
        st.write("Top Assignment Groups:")
        top_groups_df = pd.DataFrame(list(stats['top_assignment_groups'].items()), 
                                     columns=['Assignment Group', 'Count'])
        st.table(top_groups_df)
    
    # Column Statistics (below overview, expandable)
    if output_df is not None and stat_fields:
        with st.expander("Column Statistics (Click to Expand)", expanded=False):
            for col in stat_fields:
                if col in output_df.columns:
                    st.write(f"**Top Values for {col.title()}**")
                    try:
                        col_counts = output_df[col].value_counts().head(10).to_dict()
                        col_df = pd.DataFrame(list(col_counts.items()), columns=[col.title(), 'Count'])
                        st.table(col_df)
                    except Exception as e:
                        st.warning(f"Failed to generate stats for {col}: {str(e)}")
                else:
                    st.warning(f"Column '{col}' not found in output data.")
    else:
        st.warning("No output data available to generate statistics.")
    
    # Patterns with incident counts, sorted by incident count descending
    if stats['patterns_per_group']:
        st.subheader("Patterns (Top Keywords) per Assignment Group")
        for group, patterns in stats['patterns_per_group'].items():
            st.write(f"**{group}**")
            sorted_patterns = dict(sorted(patterns.items(), key=lambda x: x[1]['incident_count'], reverse=True))
            pattern_data = [
                {'Issue Type': cluster_name, 'Top Keywords': ', '.join(pattern['keywords']), 'Incident Count': pattern['incident_count']}
                for cluster_name, pattern in sorted_patterns.items()
            ]
            pattern_df = pd.DataFrame(pattern_data)
            st.table(pattern_df)
    
    # Download buttons
    col_download1, col_download2 = st.columns(2)
    with col_download1:
        if csv_data:
            st.subheader("Download Detailed CSV")
            st.download_button(
                label="Download CSV with Pattern Keywords",
                data=csv_data,
                file_name="analyzed_incidents.csv",
                mime="text/csv"
            )
    
    with col_download2:
        if output_df is not None:
            excel_data = create_excel_summary(stats, stat_fields, output_df)
            st.subheader("Download Summary Excel")
            st.download_button(
                label="Download Stats Summary (Excel)",
                data=excel_data,
                file_name="incident_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    if uploaded_file is not None:
        st.info("Click 'Review' to configure analysis options.")
    else:
        st.info("Please upload a CSV file and click 'Review' to proceed.")
