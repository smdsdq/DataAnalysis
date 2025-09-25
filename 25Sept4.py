import pandas as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from langdetect import detect_langs
import io
import csv
import warnings
import argparse
import sys
import os
import chardet
import re
from datetime import datetime

warnings.filterwarnings('ignore')

# Function to generate unique filename
def generate_unique_filename(base_name, extension, directory):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"{base_name}_{timestamp}{extension}")
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(directory, f"{base_name}_{timestamp}_{counter}{extension}")
        counter += 1
    return filename

# Function to safely detect language
def detect_language(text):
    try:
        if pd.isna(text) or str(text).strip() == '':
            return 'unknown'
        langs = detect_langs(str(text))
        if langs:
            lang = langs[0].lang
            prob = langs[0].prob
            if prob > 0.9:
                return lang if lang != 'en' else 'en'
        return 'unknown'
    except Exception:
        return 'unknown'

# Function to preprocess text, handling multiline and double quotes
def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = str(text).replace('\n', ' ').replace('\r', ' ').replace('"', '').lower().strip()
    return text

# Function to generate professional cluster names
def generate_cluster_name(keywords):
    if not keywords or len(keywords) == 0:
        return "Unclassified Issue"
    if len(keywords) >= 2:
        return f"{keywords[0].replace(',', '').title()} and {keywords[1].replace(',', '').title()} Issues"
    return f"{keywords[0].replace(',', '').title()} Issues"

# Function to preprocess CSV to handle extra fields
def preprocess_csv(input_file, is_streamlit=True):
    try:
        if is_streamlit:
            input_file.seek(0)
            content = input_file.read().decode('utf-8').splitlines()
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read().splitlines()
        
        csv_reader = csv.reader(content, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        rows = list(csv_reader)
        if not rows:
            print_function("CSV file is empty.", is_streamlit, error=True)
            return input_file
        
        headers = rows[0]
        header_count = len(headers)
        desc_idx = next((i for i, col in enumerate(headers) if col.lower().strip() in ['description', 'short_description']), None)
        if desc_idx is None:
            print_function("No 'description' or 'short_description' column found. Using first column for fixing.", is_streamlit, warning=True)
            desc_idx = 0
        
        # Check for rows with extra fields
        fixed_rows = [headers]
        for row_num, row in enumerate(rows[1:], start=2):
            if len(row) > header_count:
                print_function(f"Found row {row_num} with {len(row)} fields, expected {header_count}. Fixing description...", is_streamlit, warning=True)
                # Assume extra fields are in description; merge and replace commas/quotes
                fixed_row = row[:desc_idx]
                desc_content = ','.join(row[desc_idx:desc_idx + len(row) - header_count + 1])
                desc_content = desc_content.replace(',', '_').replace('"', "'")
                fixed_row.append(desc_content)
                fixed_row.extend(row[desc_idx + len(row) - header_count + 1:])
                fixed_rows.append(fixed_row)
            else:
                fixed_rows.append(row)
        
        # Write fixed CSV
        if is_streamlit:
            output_buffer = io.StringIO()
            csv_writer = csv.writer(output_buffer, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
            csv_writer.writerows(fixed_rows)
            output_buffer.seek(0)
            print_function("Preprocessed CSV to handle extra fields.", is_streamlit)
            return io.BytesIO(output_buffer.getvalue().encode('utf-8'))
        else:
            temp_file = 'temp_preprocessed.csv'
            with open(temp_file, 'w', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
                csv_writer.writerows(fixed_rows)
            print_function(f"Preprocessed CSV saved to '{temp_file}' for reading.", is_streamlit)
            return temp_file
    except Exception as e:
        print_function(f"Failed to preprocess CSV: {str(e)}", is_streamlit, warning=True)
        return input_file

# Function to detect encoding and convert CSV to UTF-8
def convert_to_utf8(input_file, is_streamlit=True):
    try:
        if is_streamlit:
            input_file.seek(0)
            raw_data = input_file.read()
        else:
            with open(input_file, 'rb') as f:
                raw_data = f.read()
        
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding']
        confidence = result['confidence']
        
        if detected_encoding is None or confidence < 0.5:
            print_function(f"Could not detect encoding reliably (confidence: {confidence}). Assuming utf-8.", is_streamlit, warning=True)
            detected_encoding = 'utf-8'
        
        if detected_encoding.lower() == 'utf-8':
            print_function("File is already UTF-8, no conversion needed.", is_streamlit)
            return input_file
        
        decoded_content = raw_data.decode(detected_encoding)
        output_buffer = io.BytesIO()
        output_buffer.write(decoded_content.encode('utf-8'))
        output_buffer.seek(0)
        
        print_function(f"Converted CSV from {detected_encoding} to UTF-8.", is_streamlit)
        return output_buffer if is_streamlit else output_buffer.getvalue()
    except Exception as e:
        print_function(f"Failed to convert CSV to UTF-8: {str(e)}", is_streamlit, warning=True)
        return input_file

# Function to attempt reading CSV with multiple encodings
def read_csv_with_encodings(uploaded_file, is_streamlit=True):
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'windows-1252']
    for encoding in encodings:
        try:
            if is_streamlit:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True, encoding=encoding)
                print_function(f"Successfully read CSV with encoding: {encoding}", is_streamlit)
                return df, encoding
            else:
                df = pd.read_csv(uploaded_file, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True, encoding=encoding)
                print_function(f"Successfully read CSV with encoding: {encoding}", is_streamlit)
                return df, encoding
        except UnicodeDecodeError as e:
            print_function(f"Failed to read CSV with {encoding}: {str(e)}. Trying to convert to UTF-8...", is_streamlit, warning=True)
            converted_file = convert_to_utf8(uploaded_file, is_streamlit)
            try:
                if is_streamlit:
                    converted_file.seek(0)
                    df = pd.read_csv(converted_file, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True, encoding='utf-8')
                    print_function("Successfully read converted UTF-8 CSV.", is_streamlit)
                    return df, 'utf-8'
                else:
                    temp_file = 'temp_utf8.csv'
                    with open(temp_file, 'wb') as f:
                        f.write(converted_file)
                    df = pd.read_csv(temp_file, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True, encoding='utf-8')
                    os.remove(temp_file)
                    print_function("Successfully read converted UTF-8 CSV.", is_streamlit)
                    return df, 'utf-8'
            except Exception as e2:
                print_function(f"Failed to read converted UTF-8 CSV: {str(e2)}", is_streamlit, warning=True)
        except pd.errors.ParserError as e:
            print_function(f"Parsing error with {encoding}: {str(e)}. Trying to preprocess CSV...", is_streamlit, warning=True)
            preprocessed_file = preprocess_csv(uploaded_file, is_streamlit)
            try:
                if is_streamlit:
                    preprocessed_file.seek(0)
                    df = pd.read_csv(preprocessed_file, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True, encoding='utf-8')
                    print_function("Successfully read preprocessed CSV.", is_streamlit)
                    return df, 'utf-8'
                else:
                    df = pd.read_csv(preprocessed_file, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True, encoding='utf-8')
                    os.remove(preprocessed_file)
                    print_function("Successfully read preprocessed CSV.", is_streamlit)
                    return df, 'utf-8'
            except Exception as e2:
                print_function(f"Failed to read preprocessed CSV: {str(e2)}. Trying fallback reader...", is_streamlit, warning=True)
                df = fallback_csv_reader(uploaded_file, encoding, is_streamlit)
                if df is not None:
                    print_function(f"Successfully read CSV with fallback reader using encoding: {encoding}", is_streamlit)
                    return df, encoding
        except Exception as e:
            print_function(f"Unexpected error with {encoding}: {str(e)}", is_streamlit, warning=True)
    print_function('Failed to read CSV with encodings: {}. Check file format and ensure fields with double quotes are properly quoted (e.g., "text""text").'.format(', '.join(encodings)), is_streamlit, error=True)
    return None, None

# Fallback CSV reader to handle complex cases
def fallback_csv_reader(uploaded_file, encoding, is_streamlit=True):
    try:
        if is_streamlit:
            uploaded_file.seek(0)
            content = uploaded_file.read().decode(encoding).splitlines()
        else:
            with open(uploaded_file, 'r', encoding=encoding) as f:
                content = f.read().splitlines()
        csv_reader = csv.reader(content, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        rows = list(csv_reader)
        if not rows:
            print_function("CSV file is empty.", is_streamlit, error=True)
            return None
        headers = [col.lower().strip() for col in rows[0]]
        df = pd.DataFrame(rows[1:], columns=headers)
        return df
    except Exception as e:
        print_function(f"Fallback CSV reader failed: {str(e)}", is_streamlit, warning=True)
        return None

# Main analysis function
def analyze_data(uploaded_file, max_features, n_clusters_base, ngram_max, pattern_field, stat_fields, group_field, is_streamlit=True):
    try:
        # Read CSV with encoding detection
        df, used_encoding = read_csv_with_encodings(uploaded_file, is_streamlit)
        if df is None:
            return None, None, None, None
        
        # Normalize column names to lowercase
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Validate pattern field
        if pattern_field not in df.columns:
            print_function(f"Selected pattern field '{pattern_field}' not found in CSV. Defaulting to first available column.", is_streamlit, warning=True)
            pattern_field = df.columns[0] if len(df.columns) > 0 else None
            if not pattern_field:
                print_function("No columns available in CSV for analysis.", is_streamlit, error=True)
                return None, None, None, None
        
        # Validate group field
        if group_field and group_field not in df.columns:
            print_function(f"Selected group field '{group_field}' not found. Clustering without grouping.", is_streamlit, warning=True)
            group_field = None
        
        # Validate stat fields
        stat_fields = [col for col in stat_fields if col in df.columns]
        if not stat_fields:
            print_function("No valid statistics fields selected. Proceeding without additional statistics.", is_streamlit, warning=True)
        
        # Ensure requested columns are included
        required_columns = ['opened_at', 'state', 'assigned_to']
        for col in required_columns:
            if col not in df.columns:
                print_function(f"Column '{col}' not found in input CSV.", is_streamlit, warning=True)
        
        # Handle large dataset
        if len(df) > 1000000:
            print_function(f"Dataset has {len(df)} rows. Processing all, but clustering may take time.", is_streamlit, warning=True)
        
        # Detect languages and filter English
        print_function("Detecting languages...", is_streamlit)
        df['detected_lang'] = df[pattern_field].apply(detect_language)
        english_mask = df['detected_lang'] == 'en'
        analyzed_df = df[english_mask].copy()
        omitted_count = len(df) - len(analyzed_df)
        analyzed_count = len(analyzed_df)
        
        if analyzed_count == 0:
            print_function("No English records found. Proceeding with all records.", is_streamlit, warning=True)
            analyzed_df = df.copy()
        
        # Classify resolutions based on assignee
        assignee_col = next((col for col in df.columns if col.lower() == 'assignee'), None)
        if assignee_col:
            df['resolution_type'] = df[assignee_col].apply(
                lambda x: 'Automated' if pd.notna(x) and re.search(r'VW', str(x), re.IGNORECASE) else 'Manual'
            )
            automated_count = len(df[df['resolution_type'] == 'Automated'])
            manual_count = len(df[df['resolution_type'] == 'Manual'])
        else:
            print_function("No 'assignee' column found. Skipping resolution type classification.", is_streamlit, warning=True)
            automated_count = 0
            manual_count = len(df)
        
        # Preprocess text for pattern analysis
        analyzed_df['analysis_text'] = analyzed_df[pattern_field].apply(preprocess_text)
        
        # Group by group_field if provided
        if group_field:
            groups = analyzed_df.groupby(group_field)
        else:
            groups = [(None, analyzed_df)]
        
        # Prepare output DF
        output_cols = [col for col in analyzed_df.columns if col != 'detected_lang']
        output_df = analyzed_df[output_cols].copy()
        output_df['pattern_keywords'] = np.nan
        
        # Perform clustering
        print_function("Performing clustering...", is_streamlit)
        cluster_results = {}
        for group_name, group_data in groups:
            group_name = str(group_name) if group_name is not None else "All Records"
            if len(group_data) < 2:
                print_function(f"Skipping group '{group_name}' with fewer than 2 records.", is_streamlit, warning=True)
                continue
            
            texts = group_data['analysis_text'].fillna('').tolist()
            
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, ngram_max))
            try:
                X = vectorizer.fit_transform(texts)
                
                # Clustering with MiniBatchKMeans
                n_clusters = min(n_clusters_base, len(group_data) // 10)
                if n_clusters < 2:
                    n_clusters = 2
                
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
                clusters = kmeans.fit_predict(X)
                
                # Extract top keywords and incident counts
                patterns = {}
                feature_names = vectorizer.get_feature_names_out()
                for i in range(n_clusters):
                    cluster_mask = clusters == i
                    cluster_size = np.sum(cluster_mask)
                    cluster_terms = X[cluster_mask].mean(axis=0).A1
                    top_indices = np.argsort(cluster_terms)[-5:][::-1]
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
                print_function(f"Clustering failed for group '{group_name}': {str(e)}", is_streamlit, warning=True)
                output_df.loc[group_data.index, 'pattern_keywords'] = 'clustering_failed'
        
        # High-level stats
        stats = {
            'total_records': len(df),
            'analyzed_records': analyzed_count,
            'omitted_records': omitted_count,
            'non_english_languages': df['detected_lang'].value_counts().to_dict(),
            'patterns_per_group': cluster_results,
            'automated_count': automated_count,
            'manual_count': manual_count
        }
        if group_field:
            stats['top_groups'] = analyzed_df[group_field].value_counts().head(10).to_dict()
        
        # Prepare CSV buffer
        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        csv_data = csv_buffer.getvalue()
        
        return stats, csv_data, output_df, list(analyzed_df.columns)
        
    except Exception as e:
        print_function('An error occurred during analysis: {}. Available columns: {}'.format(str(e), list(df.columns) if 'df' in locals() else 'Not loaded'), is_streamlit, error=True)
        return None, None, None, None

# Function to create Excel summary
def create_excel_summary(stats, stat_fields, output_df, group_field):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Dataset Overview sheet
        overview_data = {
            'Metric': ['Total Records', 'Analyzed (English)', 'Omitted (Non-English/Unknown)', 'Automated Resolutions', 'Manual Resolutions'],
            'Value': [stats['total_records'], stats['analyzed_records'], stats['omitted_records'], stats['automated_count'], stats['manual_count']]
        }
        pd.DataFrame(overview_data).to_excel(writer, sheet_name='Overview', index=False)
        
        # Non-English Languages sheet
        if stats['non_english_languages']:
            non_eng_df = pd.DataFrame(list(stats['non_english_languages'].items()), columns=['Language', 'Count'])
            non_eng_df.to_excel(writer, sheet_name='Non-English Languages', index=False)
        
        # Top Groups sheet
        if group_field and 'top_groups' in stats:
            top_groups_df = pd.DataFrame(list(stats['top_groups'].items()), columns=[group_field.title(), 'Count'])
            top_groups_df.to_excel(writer, sheet_name='Top Groups', index=False)
        
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
            safe_group_name = group.replace('/', '_').replace('\\', '_')[:31]
            pattern_df.to_excel(writer, sheet_name=f'{safe_group_name} Patterns', index=False)
    
    return output.getvalue()

# Helper function to print messages
def print_function(message, is_streamlit=True, warning=False, error=False):
    if is_streamlit:
        if error:
            st.error(message)
        elif warning:
            st.warning(message)
        else:
            st.info(message)
    else:
        prefix = "ERROR: " if error else "WARNING: " if warning else ""
        print(f"{prefix}{message}")

# CLI execution function
def run_cli(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(args.csv_file):
        print(f"ERROR: CSV file '{args.csv_file}' not found.")
        sys.exit(1)
    
    stats, csv_data, output_df, available_columns = analyze_data(
        args.csv_file,
        args.max_features,
        args.n_clusters_base,
        args.ngram_max,
        args.pattern_field,
        args.stat_fields,
        args.group_field,
        is_streamlit=False
    )
    
    if stats is None:
        print("Analysis failed. Check error messages above.")
        sys.exit(1)
    
    # Print Dataset Overview
    print("\nDataset Overview:")
    print(f"Total Records: {stats['total_records']}")
    print(f"Analyzed (English): {stats['analyzed_records']}")
    print(f"Omitted (Non-English/Unknown): {stats['omitted_records']}")
    print(f"Automated Resolutions: {stats['automated_count']}")
    print(f"Manual Resolutions: {stats['manual_count']}")
    
    if stats['non_english_languages']:
        print("\nTop Non-English Languages:")
        non_eng_df = pd.DataFrame(list(stats['non_english_languages'].items()), columns=['Language', 'Count']).head(5)
        print(non_eng_df.to_string(index=False))
    
    if args.group_field and 'top_groups' in stats:
        print(f"\nTop {args.group_field.title()}:")
        top_groups_df = pd.DataFrame(list(stats['top_groups'].items()), columns=[args.group_field.title(), 'Count'])
        print(top_groups_df.to_string(index=False))
    
    # Print Column Statistics
    if output_df is not None and args.stat_fields:
        print("\nColumn Statistics:")
        for col in args.stat_fields:
            if col in output_df.columns:
                print(f"\nTop Values for {col.title()}:")
                try:
                    col_counts = output_df[col].value_counts().head(10).to_dict()
                    col_df = pd.DataFrame(list(col_counts.items()), columns=[col.title(), 'Count'])
                    print(col_df.to_string(index=False))
                except Exception as e:
                    print(f"WARNING: Failed to generate stats for {col}: {str(e)}")
            else:
                print(f"WARNING: Column '{col}' not found in output data.")
    
    # Print Patterns
    if stats['patterns_per_group']:
        print("\nPatterns (Top Keywords) per Group:")
        for group, patterns in stats['patterns_per_group'].items():
            print(f"\n{group}:")
            sorted_patterns = dict(sorted(patterns.items(), key=lambda x: x[1]['incident_count'], reverse=True))
            pattern_data = [
                {'Issue Type': cluster_name, 'Top Keywords': ', '.join(pattern['keywords']), 'Incident Count': pattern['incident_count']}
                for cluster_name, pattern in sorted_patterns.items()
            ]
            pattern_df = pd.DataFrame(pattern_data)
            print(pattern_df.to_string(index=False))
    
    # Save outputs with unique filenames
    if csv_data:
        csv_filename = generate_unique_filename("analyzed_incidents", ".csv", script_dir)
        with open(csv_filename, "w", encoding='utf-8') as f:
            f.write(csv_data)
        print(f"\nSaved detailed CSV to '{csv_filename}'")
    
    if output_df is not None:
        excel_data = create_excel_summary(stats, args.stat_fields, output_df, args.group_field)
        excel_filename = generate_unique_filename("incident_summary", ".xlsx", script_dir)
        with open(excel_filename, "wb") as f:
            f.write(excel_data)
        print(f"Saved summary Excel to '{excel_filename}'")

# Streamlit UI
def run_streamlit():
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
    if 'group_field' not in st.session_state:
        st.session_state.group_field = None

    # Step 2: Column selection
    if uploaded_file is not None and review_button:
        try:
            df, _ = read_csv_with_encodings(uploaded_file, is_streamlit=True)
            if df is not None:
                df.columns = [col.lower().strip() for col in df.columns]
                st.session_state.df = df
                st.session_state.available_columns = list(df.columns)
                group_candidates = ['assignment_group', 'group', 'assigned_to_group', 'category', 'sub_category', 'subcategory']
                st.session_state.group_field = next((col for col in group_candidates if col in df.columns), df.columns[0] if df.columns.size > 0 else None)
            else:
                st.error("Failed to load CSV. Please check the file and try again.")
        except Exception as e:
            st.error(f'Failed to read CSV: {str(e)}. Ensure fields with double quotes are escaped (e.g., "text""text") and file format is valid.')
    
    if st.session_state.available_columns:
        st.header("Step 2: Configure Analysis")
        
        # Clustering parameters
        st.sidebar.header("Clustering Parameters")
        max_features = st.sidebar.slider(
            "Max TF-IDF Features", 500, 5000, 1000, step=500,
            help="Controls the vocabulary size for text features. Higher values capture more details but increase computation time."
        )
        n_clusters_base = st.sidebar.slider(
            "Base Number of Clusters per Group", 2, 20, 5,
            help="Maximum clusters per group. Higher values create more specific clusters but may overfit small groups."
        )
        ngram_max = st.sidebar.selectbox(
            "Max N-gram Range", [1, 2, 3], index=1,
            help="Maximum length of word phrases (n-grams). 1: single words; 2: up to bigrams (e.g., 'login failure'); 3: up to trigrams."
        )
        
        # Column selection
        st.subheader("Select Fields")
        pattern_options = st.session_state.available_columns
        if not pattern_options:
            st.error("No columns available in CSV for pattern analysis.")
            pattern_field = None
        else:
            default_pattern = next((col for col in ['description', 'short_description', 'desc', 'short desc', 'summary', 'long_description'] if col in pattern_options), pattern_options[0])
            pattern_field = st.selectbox(
                "Field for Pattern Analysis",
                options=pattern_options,
                index=pattern_options.index(default_pattern) if default_pattern in pattern_options else 0
            )
        
        stat_fields = st.multiselect(
            "Fields for Statistics",
            options=pattern_options,
            default=[col for col in ['category', 'sub_category', 'assignment_group'] if col in pattern_options]
        )
        st.session_state.stat_fields = stat_fields
        
        group_field = st.selectbox(
            "Field for Grouping Clusters",
            options=['None'] + pattern_options,
            index=pattern_options.index(st.session_state.group_field) + 1 if st.session_state.group_field in pattern_options else 0
        )
        group_field = None if group_field == 'None' else group_field
        st.session_state.group_field = group_field
        
        if pattern_field:
            analyze_button = st.button("Start Analysis")
            if analyze_button:
                with st.spinner("Analyzing data..."):
                    stats, csv_data, output_df, available_columns = analyze_data(
                        uploaded_file, max_features, n_clusters_base, ngram_max, pattern_field, stat_fields, group_field, is_streamlit=True
                    )
                    st.session_state.stats = stats
                    st.session_state.output_df = output_df
                    st.session_state.csv_data = csv_data
                    st.session_state.available_columns = available_columns if available_columns else st.session_state.available_columns
        else:
            st.warning("Please select a valid pattern field to proceed with analysis.")
    
    # Display results
    if st.session_state.stats:
        stats = st.session_state.stats
        output_df = st.session_state.output_df
        csv_data = st.session_state.csv_data
        stat_fields = st.session_state.stat_fields
        group_field = st.session_state.group_field
        
        # Dataset Overview
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Total Records: {stats['total_records']}")
            st.write(f"Analyzed (English): {stats['analyzed_records']}")
            st.write(f"Omitted (Non-English/Unknown): {stats['omitted_records']}")
            st.write(f"Automated Resolutions: {stats['automated_count']}")
            st.write(f"Manual Resolutions: {stats['manual_count']}")
            
            if stats['non_english_languages']:
                st.write("Top Non-English Languages:")
                non_eng_df = pd.DataFrame(list(stats['non_english_languages'].items()), 
                                          columns=['Language', 'Count']).head(5)
                st.table(non_eng_df)
        
        with col2:
            if group_field and 'top_groups' in stats:
                st.write(f"Top {group_field.title()}:")
                top_groups_df = pd.DataFrame(list(stats['top_groups'].items()), 
                                             columns=[group_field.title(), 'Count'])
                st.table(top_groups_df)
        
        # Column Statistics
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
            st.info("No statistics fields selected or no output data available.")
        
        # Patterns
        if stats['patterns_per_group']:
            st.subheader("Patterns (Top Keywords) per Group")
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
                excel_data = create_excel_summary(stats, stat_fields, output_df, group_field)
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

# Main execution
if __name__ == "__main__":
    # Check if running as Streamlit or CLI
    if 'streamlit' in sys.modules and hasattr(st, 'script_run_context'):
        run_streamlit()
    else:
        parser = argparse.ArgumentParser(description="ServiceNow Incident Analysis Tool")
        parser.add_argument('--csv_file', type=str, required=True, help="Path to the input CSV file")
        parser.add_argument('--max_features', type=int, default=1000, help="Max TF-IDF features (500-5000)")
        parser.add_argument('--n_clusters_base', type=int, default=5, help="Base number of clusters per group (2-20)")
        parser.add_argument('--ngram_max', type=int, default=2, choices=[1, 2, 3], help="Max n-gram range (1-3)")
        parser.add_argument('--pattern_field', type=str, default='description', help="Column for pattern analysis (default: description)")
        parser.add_argument('--stat_fields', type=str, default=None, help="Comma-separated columns for statistics (e.g., category,sub_category)")
        parser.add_argument('--group_field', type=str, default=None, help="Column for grouping clusters (optional)")
        args = parser.parse_args()
        
        args.stat_fields = args.stat_fields.split(',') if args.stat_fields else []
        run_cli(args)
