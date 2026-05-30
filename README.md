Inbound - Enterprise Email Spam Detection Framework

An industry-grade, production-ready machine learning framework and interactive SaaS diagnostics dashboard that detects malicious spam signatures using natural language processing (NLP).

Performance Matrix & Core Architecture

This project maps raw categorical communication records directly down into normalized tokens, which are run against pre-fitted vectors using a high-throughput Multinomial Naive Bayes Classification runtime.

Overall Accuracy: `~98%` (Baseline validation accuracy derived against stratified 33% test dataset bounds)
feature Pipeline Architecture:Clean Text Constraints -> Token Vector Extraction Matrix -> Multinomial Naive Bayes Scoring Logic.



 Structural Overview

```text
Email-Spam-Detection/
│
├── data/
│   └── spam.csv                # Native Raw Input Dataset Source Files
│
├── notebooks/
│   └── model_training.ipynb    # Historical Scoping Sandbox Notebooks
│
├── models/
│   ├── spam_model.pkl          # Exported Serialized Classifier Binary Binaries
│   └── vectorizer.pkl          # Feature Extraction Serialization Pipeline
│
├── src/
│   ├── preprocess.py           # Object-Oriented NLP Text Data Cleansing Core Functions
│   ├── train.py                # Model Orchestration Pipeline Orchestration Routines 
│   └── predict.py              # Single or Bulk Inference Prediction Frameworks
│
├── app.py                      # Premium Streamlit UI Dashboard Interface Code
├── requirements.txt            # System Manifest Dependencies
├── README.md                   # Technical Documentation Manifest
└── .gitignore                  # Development Track Ignores Rules