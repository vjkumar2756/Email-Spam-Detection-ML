import os
import sys
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import TextPreprocessor

def run_training_pipeline(data_path: str, output_model_dir: str = 'models'):
    """
    Loads raw CSV data, transforms features via modular preprocessing wrappers,
    fits a vectorizer and structural classifier, and serializes assets to disk.
    """
    logging.info(f"Initiating model training workflow using data source: {data_path}")
    
  
    try:
        df = pd.read_csv(data_path, encoding='latin1')
    except Exception as e:
        logging.error(f"Failed to read data file: {str(e)}")
        raise e

   
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.process_dataframe(df)
    
    X = processed_df['cleaned_message']
    y = processed_df['label']
    
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )
    
    logging.info("Vectorizing text inputs via CountVectorizer token generation.")
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    logging.info("Fitting Multinomial Naive Bayes structural estimator.")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    

    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    logging.info(f"Training cycle finalized successfully. Overall Accuracy: {accuracy:.4f}")
    

    os.makedirs(output_model_dir, exist_ok=True)
    
    model_out_path = os.path.join(output_model_dir, 'spam_model.pkl')
    vec_out_path = os.path.join(output_model_dir, 'vectorizer.pkl')
    
    joblib.dump(model, model_out_path)
    joblib.dump(vectorizer, vec_out_path)
    
    logging.info(f"Artifacts successfully exported: {model_out_path}, {vec_out_path}")
    return report

if __name__ == "__main__":
    
    data_file_path = os.path.join('data', 'spam.csv')
    if os.path.exists(data_file_path):
        run_training_pipeline(data_file_path)
    else:
        logging.error(f"Data file missing at standard path: {data_file_path}. Skipping automatic initialization routine.")