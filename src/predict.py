import os
import sys
import joblib
from typing import Dict, Any, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import TextPreprocessor

class SpamPredictor:
    """
    Production interface pipeline managing predictions using precompiled disk artifacts.
    """
    def __init__(self, model_dir: str = 'models'):
        model_path = os.path.join(model_dir, 'spam_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Precompiled model or vectorizer binaries are missing. Please execute train.py first.")
            
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.preprocessor = TextPreprocessor()

    def predict(self, raw_message: str) -> Dict[str, Any]:
        """
        Parses raw text inputs and runs inferences against the trained ML model.
        Returns a rich metadata dictionary containing scores, tokens, and classifications.
        """
        cleaned_text = self.preprocessor.clean_text(raw_message)
        
        
        if not cleaned_text.strip():
            return {
                "label": "HAM",
                "label_code": 0,
                "probability_spam": 0.0,
                "probability_ham": 1.0,
                "cleaned_text": ""
            }
            
        vectorized_input = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vectorized_input)[0]
        probabilities = self.model.predict_proba(vectorized_input)[0]
        
        return {
            "label": "SPAM" if prediction == 1 else "HAM",
            "label_code": int(prediction),
            "probability_spam": float(probabilities[1]),
            "probability_ham": float(probabilities[0]),
            "cleaned_text": cleaned_text
        }