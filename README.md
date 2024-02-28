# twitter

import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        import warnings
warnings.filterwarnings("ignore")
warnings.resetwarnings()
warnings.filterwarnings("ignore", category=DeprecationWarning)
train = pd.read_csv('/content/twitter_training.csv', names = ['id','source','sentiment','tweet'])
test = pd.read_csv('/content/twitter_validation.csv', names = ['id','source','sentiment','tweet'])
train.head()
df = train
df.info()
df.describe()
df.isnull().sum()
df = df.dropna(axis=0)
df.isnull().sum()
df.info()
test.isnull().sum()
test = test.dropna(axis=0)
import re

def clean_text(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    
    # Remove mentions and hashtags
    tweet = re.sub(r'@[A-Za-z0-9_]+|#[A-Za-z0-9_]+', '', tweet)
    
    # Remove special characters, numbers, and punctuation
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)
    
    # Remove 'RT' (Retweet) indicator
    tweet = re.sub(r'\bRT\b', '', tweet)
    
    return tweet.lower()
    df.loc[: ,'tweet'] = df['tweet'].apply(clean_text)
df.head()
test.loc[:,'tweet'] = test['tweet'].apply(clean_text)
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
def remove_stopwords(tweet):
    words = tweet.split(' ')
    words = [word for word in words if word not in set(stopwords.words('english'))]
    tweet = ' '.join(words)
    return tweet
    df.loc[:,'tweet'] = df['tweet'].apply(remove_stopwords)
df.head()
test.loc[:,'tweet'] = test['tweet'].apply(remove_stopwords)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
models = [
    MultinomialNB(), LogisticRegression(),
    RandomForestClassifier(n_jobs=-1),
    GradientBoostingClassifier()]

# Feature extraction methods
vectorizers = [ ('TF-IDF', TfidfVectorizer()),
    ('Count Vectorizer', CountVectorizer())
]X_train = df['tweet']
y_train = df['sentiment']
X_test = test['tweet']
y_test = test['sentiment']for model in models:
    for vec_name, vec in vectorizers:
        pipeline = Pipeline([
            ('vectorizer',vec),( 'classifier', model)
        ])
        
        pipeline.fit(X_train,y_train)
        
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)
        cf = confusion_matrix(y_test,y_pred)
        
        print(f"\nModel: {model.__class__.__name__}, Vectorizer: {vec_name}")
        print("Confusion Matrix:\n", cf)
        print("Classification Report:\n", report)
