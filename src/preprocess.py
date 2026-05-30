import string
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Ensure necessary NLTK components are safely downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    """
    Production-grade text preprocessing pipeline for NLP feature extraction.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Create a translation table for removing punctuation efficiently
        self.punctuation_table = str.maketrans("", "", string.punctuation)

    def clean_text(self, text: str) -> str:
        """
        Applies a clean sequence of textual transformations to normalize input data:
        Lowercase -> Punctuation Removal -> Digit Removal -> Non-ASCII (Emoji) Removal -> Stopword Removal.
        """
        if not isinstance(text, str):
            return ""

        # 1. Lowercase normalization
        text = text.lower()
        
        # 2. Punctuation removal
        text = text.translate(self.punctuation_table)
        
        # 3. Digit stripping
        text = re.sub(r'\d+', '', text)
        
        # 4. Remove non-ASCII characters (emojis, special icons)
        text = text.encode("ascii", "ignore").decode("utf-8")
        
        # 5. Tokenization & Stopword filtering
        words = text.split()
        cleaned_words = [word for word in words if word not in self.stop_words]
        
        return " ".join(cleaned_words)

    def process_dataframe(self, df: pd.DataFrame, message_col: str = 'message', label_col: str = 'label') -> pd.DataFrame:
        """
        Transforms a raw Pandas dataframe into structural features prepared for ML tasks.
        """
        processed_df = df.copy()
        
    
        if 'v1' in processed_df.columns and 'v2' in processed_df.columns:
            processed_df = processed_df[['v1', 'v2']]
            processed_df.columns = [label_col, message_col]

    
        processed_df[message_col] = processed_df[message_col].astype(str)
        if processed_df[label_col].dtype == object:
            processed_df[label_col] = processed_df[label_col].map({'ham': 0, 'spam': 1})
            
      
        processed_df = processed_df.dropna(subset=[label_col, message_col])
        
      
        processed_df[f'cleaned_{message_col}'] = processed_df[message_col].apply(self.clean_text)
        return processed_df