import pandas as pd
import os
import glob
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_combined_data(data_directory):
    """Reads all CSV files from a directory and combines them into a single DataFrame."""
    if not os.path.isdir(data_directory):
        logging.error(f"Directory not found: {data_directory}")
        return None

    all_files = glob.glob(os.path.join(data_directory, '*.csv'))
    if not all_files:
        logging.warning(f"No CSV files found in directory: {data_directory}")
        return None

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            logging.info(f"Read file: {os.path.basename(file)} with {len(df)} rows.")
        except pd.errors.EmptyDataError:
            logging.warning(f"Skipping empty file: {os.path.basename(file)}")
        except Exception as e:
            logging.error(f"Error reading {os.path.basename(file)}: {e}")

    if not df_list:
        logging.error("No valid data found in any CSV file.")
        return None

    return pd.concat(df_list, ignore_index=True)

def train_or_load_model(combined_df, model_path='ticket_classifier_model.joblib'):
    """Trains a new model or loads an existing one from a file."""
    if os.path.exists(model_path):
        logging.info(f"Loading pre-trained model from '{model_path}'.")
        return joblib.load(model_path)

    if 'description' not in combined_df.columns or 'category' not in combined_df.columns:
        logging.error("Combined data is missing 'description' or 'category' columns.")
        return None

    logging.info("Starting model training and parameter tuning...")
    X_train, _, y_train, _ = train_test_split(combined_df['description'], combined_df['category'], test_size=0.2, random_state=42)

    # Define the pipeline and parameter grid for Grid Search
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_df': [0.75, 1.0],
        'classifier__alpha': [0.1, 1.0]
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best model parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # Save the trained model
    joblib.dump(best_model, model_path)
    logging.info(f"Model trained and saved to '{model_path}'.")
    return best_model

def analyze_and_report(combined_df, model, excel_file_name='ticket_analysis_results.xlsx'):
    """Performs analysis and writes the results to an Excel file."""
    if combined_df is None or model is None:
        logging.error("Cannot perform analysis due to missing data or model.")
        return

    logging.info("Making predictions on the full dataset.")
    combined_df['predicted_category'] = model.predict(combined_df['description'])
    
    # Generate the summary report
    summary_df = combined_df.groupby('predicted_category').size().reset_index(name='number_of_tickets')
    logging.info("Generated summary report of ticket groupings.")

    # Write results to Excel
    try:
        with pd.ExcelWriter(excel_file_name, engine='xlsxwriter') as writer:
            combined_df.to_excel(writer, sheet_name='All Tickets', index=False)
            summary_df.to_excel(writer, sheet_name='Tickets Grouped', index=False)
        logging.info(f"Analysis successfully saved to '{excel_file_name}' with two sheets.")
    except Exception as e:
        logging.error(f"Failed to write to Excel file: {e}")

if __name__ == '__main__':
    # --- Data Simulation (Remove this section if using your own data) ---
    temp_dir = 'it_tickets_data'
    os.makedirs(temp_dir, exist_ok=True)
    pd.DataFrame({'description': ["My laptop is slow"], 'category': ["Hardware"]}).to_csv(os.path.join(temp_dir, 'sim_data.csv'), index=False)
    # --- End of Simulation ---

    data_directory = 'it_tickets_data'
    combined_df = get_combined_data(data_directory)

    if combined_df is not None and not combined_df.empty:
        model = train_or_load_model(combined_df)
        if model is not None:
            analyze_and_report(combined_df, model)
            logging.info("Script finished successfully.")
    else:
        logging.error("Script aborted due to no data.")
