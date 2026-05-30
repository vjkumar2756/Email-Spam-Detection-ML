 Email Spam Detection Using NLP and Machine Learning

 Project Overview

Email spam detection is a Natural Language Processing (NLP) application that automatically classifies emails as **Spam** or **Ham (Not Spam)**. This project uses machine learning techniques and text preprocessing methods to build an intelligent spam filtering system capable of identifying unwanted emails with high accuracy.

The goal of this project is to improve email security, reduce phishing attempts, and enhance user experience by filtering irrelevant messages automatically.


 Objectives

* Detect spam emails automatically.
* Apply NLP techniques for text preprocessing.
* Convert textual data into numerical features.
* Train and evaluate multiple machine learning models.
* Compare model performance using standard evaluation metrics.


 Dataset

Dataset Source: Kaggle Spam Email Dataset

 Features

| Column | Description           |
| ------ | --------------------- |
| label  | Spam or Ham           |
| text   | Email message content |

 Target Variable

0 → Ham (Not Spam)**
1 → Spam**

---

 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* NLTK
* Matplotlib
* Streamlit



 NLP Pipeline

1. Data Cleaning

* Remove null values
* Remove duplicate records

2. Text Preprocessing

* Convert text to lowercase
* Remove punctuation
* Remove special characters
* Remove stopwords
* Tokenization
* Stemming using Porter Stemmer

 Example

Original Text:

Congratulations! You have won a FREE lottery worth $1000.

Processed Text:

congratul free lotteri worth



Feature Engineering

Bag of Words (BoW)

Converts text into numerical vectors based on word frequency.

TF-IDF Vectorization

Assigns importance scores to words based on their frequency across documents.



 Machine Learning Models
The following models were trained and evaluated:
* Multinomial Naive Bayes
* Logistic Regression
* Decision Tree Classifier



Best Performing Model
Multinomial Naive Bayes + TF-IDF

Reasons:

* Fast training time
* High accuracy
* Excellent performance on text classification tasks
* Lightweight and easy to deploy


 Streamlit Web Application

The project includes a Streamlit-based web application where users can:

* Enter email text
* Predict Spam or Ham instantly
* View prediction confidence


Run the application:
streamlit run app.py



 Project Structure


Email-Spam-Detection/
│
├── data/
│   └── spam.csv
│
├── notebooks/
│   └── Spam_Detection.ipynb
│
├── models/
│   └── spam_model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore



Future Improvements

* Deep Learning (LSTM, GRU)
* Transformer Models (BERT)
* Multi-language Spam Detection
* Real-Time Email Integration
* Explainable AI Dashboard

---

Author 

#Vijay Fulwariya

Machine Learning & NLP Enthusiast

---

Conclusion

This project demonstrates how Natural Language Processing and Machine Learning can be combined to build an effective email spam detection system. By leveraging text preprocessing, feature engineering, and classification algorithms, the model successfully identifies spam emails and provides a practical solution for real-world email filtering applications.
