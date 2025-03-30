# AI Sentiment Analysis Project

## Overview
This project aims to determine public sentiment towards artificial intelligence (AI) on social media platforms, specifically Reddit. The project involves collecting, preprocessing, and analyzing Reddit posts to classify them into three sentiment categories: **positive, negative, and neutral**. The final objective is to build a supervised learning model to predict sentiment based on these classifications.

## Table of Contents
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Results and Evaluation](#results-and-evaluation)
- [Tools and Technologies](#tools-and-technologies)
- [Considerations](#considerations)
- [Key Terms and Definitions](#key-terms-and-definitions)
- [Conclusion](#conclusion)

## Data Collection
### Libraries Used
- `praw`: Python Reddit API Wrapper for interacting with Reddit's API
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical operations

### Authentication
```python
import praw

reddit = praw.Reddit(
    client_id='your_client_id',
    client_secret='your_client_secret',
    user_agent='your_user_agent',
    username='your_username',
    password='your_password'
)
```

### Functions
- `search_posts(query)`: Searches for Reddit posts based on a query and returns post features.
- `accumulate_data(list_queries, label)`: Collects data for queries, labels them, and saves to CSV.

### Queries for Sentiment Analysis
- **AI_IS_GOOD_QUERIES**: Queries related to positive sentiments about AI.
- **AI_IS_BAD_QUERIES**: Queries related to negative sentiments about AI.
- **NEUTRAL_QUERIES_ON_AI**: Queries related to neutral sentiments about AI.

## Data Preprocessing
### Libraries Used
- `pandas`, `numpy`, `re` (Regular Expressions)
- `nltk`: Natural Language Toolkit (stopwords, stemming)
- `sklearn`: Machine learning library for feature extraction and modeling

### Text Preprocessing Steps
1. Remove non-alphabetic characters
2. Convert text to lowercase
3. Remove stopwords
4. Apply stemming

### Feature Extraction
- `CountVectorizer`: Converts text into numerical vectors for machine learning

### Data Splitting
- Uses `train_test_split` to divide data into training and testing sets

## Model Building
### Models Tested
- **Naive Bayes**: Accuracy = #73% accuracy
- **Random Forest**: Accuracy = 77% accuracy
- **Decision Tree**: Accuracy = 73.5% accuracy

### Metrics
- **Confusion Matrix**: Evaluates model performance
- **F1 Score**: Measures precision-recall balance

## Results and Evaluation
### Evaluation Metrics
- **Confusion Matrix**: Understands model performance across different classes
- **F1 Score**: Measures model classification accuracy

### Final Data Outputs
- `AI_GOOD_TRAINING.csv`
- `AI_BAD_TRAINING.csv`
- `AI_NEUTRAL_TRAINING.csv`
- `TRAINING.csv`

## Tools and Technologies
### APIs
- **Reddit API**: For data collection

### JSON Files
- Used for lightweight data storage and transmission

### API Endpoints
- Specific URLs for accessing Reddit data

## Considerations
### Source Choice
- **Reddit**: Provides candid opinions but lacks verified users and relies on upvote/downvote systems.

### Data Cleaning & Preprocessing
- Handles missing values and ensures balanced data for training

## Key Terms and Definitions
- **API (Application Programming Interface)**: Set of functions for interacting with software applications
- **JSON (JavaScript Object Notation)**: Format for transmitting data in a structured manner
- **API Endpoints**: Specific URLs for accessing functionalities of an API

## Conclusion
This project explores public sentiment towards AI using Reddit posts. By collecting, preprocessing, and analyzing data, a supervised learning model is built to classify sentiments effectively. The choice of Reddit as a data source and the use of various machine learning tools provide a comprehensive approach to sentiment analysis.

---
### Future Enhancements
- Fine-tuning model hyperparameters
- Expanding dataset for better generalization
- Experimenting with deep learning approaches for sentiment analysis

### How to Contribute
Feel free to fork this repository and submit pull requests with improvements or additional features.

### Contact
For any queries, reach out at [your_email@example.com](asivaprakash23@gmail.com).
