import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def export_data(X_train, X_test, y_train, y_test):
    train = pd.DataFrame({'reviewText': X_train, 'rating': y_train})
    test = pd.DataFrame({'reviewText': X_test, 'rating': y_test})
    train.to_csv("train_data.csv")
    test.to_csv("test_data.csv")

def removeNums(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) # remove numbers
    text = re.sub(r'_', '', text) # remove underscores
    return text

def preprocess(path):
    df = pd.read_json(path, lines=True, dtype={'overall': np.int64}) # cast "overall" (rating) column to float
    X, y = list(df['reviewText']), list(df['overall'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67) # train test split with 2/3 train
    vectorizer = CountVectorizer(preprocessor=removeNums)
    V = vectorizer.fit_transform(X_train)
    print(vectorizer.get_feature_names())

def main():
    preprocess("./reviews_Amazon_Instant_Video_5.json")

if __name__ == "__main__":
    main()