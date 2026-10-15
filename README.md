# Email / SMS Spam Detection

Oasis Infobyte Internship (OIBSIP) | Data Science | Task 4

## Objective
Build an NLP classifier that can tell whether a message is spam or ham (legitimate), using text preprocessing, TF-IDF features and machine learning models.

## Dataset
SMS Spam Collection dataset (Kaggle / UCI), included as `spam.csv`. After removing duplicates it has 5,171 messages: about 87% ham and 13% spam.

## Steps Performed
1. Loaded the data (`latin-1` encoding) and checked the shape, nulls and duplicates
2. Removed duplicates and checked the class distribution (spam vs ham counts and percentage)
3. Text preprocessing: lowercase, removed punctuation and numbers, removed stopwords, stemming
4. Visualisations: class distribution, message length, WordClouds for spam and ham words
5. Feature extraction with TF-IDF (explained in the notebook)
6. Stratified 80/20 train/test split
7. Trained Multinomial Naive Bayes, Logistic Regression and Linear SVM
8. Evaluated with accuracy, precision, recall, F1-score and confusion matrices
9. Discussed why recall matters for spam detection

## Tools Used
Python, pandas, NumPy, scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, SVM), NLTK (Porter stemmer), WordCloud, matplotlib, seaborn, Jupyter Notebook

## Outcome
**Linear SVM** was the best model with **98.6% accuracy, 99.2% precision, 89.3% recall and 94.0% F1-score**. Multinomial Naive Bayes reached 97.2% accuracy and Logistic Regression 96.3%. All models had very high precision, but the SVM caught the most spam.

## How to Run
```bash
pip install pandas numpy scikit-learn nltk wordcloud matplotlib seaborn jupyter
jupyter notebook Spam_Detection.ipynb
```

## Demo Video
<LinkedIn link> https://lnkd.in/p/gvSCNk5n
