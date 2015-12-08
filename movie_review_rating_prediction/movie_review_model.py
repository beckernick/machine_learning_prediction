# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:35:14 2015

@author: nickbecker
"""
from __future__ import division, print_function
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Simple Bag of Words Model
train = pd.read_csv("/users/nickbecker/Python Projects/imdb_data/labeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/users/nickbecker/Python Projects/imdb_data/testData.tsv",
                    header=0, delimiter="\t", quoting=3)
train.shape
train.columns.values

print(train["review"][0])
print(train["review"][20])
print(train["sentiment"][1:20])


# Function to clean all the reviews
def clean_reviews(raw_review):
    # Removing HTML tags
    review_text_soup = BeautifulSoup(raw_review).get_text()
    
    # Dealing with Punctuation, numbers, stopwords, etc.
    letters_only = re.sub("[^a-zA-z]",
                          " ",
                          review_text_soup)
    # Split the lowercase text into a list of words
    words = letters_only.lower().split()
    stopwords_eng = set(stopwords.words("english")) # conver to set for backend speed improvement
    
    # Remove stopwords    
    useful_words = [x for x in words if not x in stopwords_eng]
    
    # Stemming and Lemmatizing

    # Combine words into a paragraph again
    useful_words_string = " ".join(useful_words)
    return(useful_words_string)


print(clean_reviews(train["review"][0]))



# Clean the training set
review_count = train["review"].size
clean_train_reviews = []

for i in xrange(0, review_count):
    clean_train_reviews.append(clean_reviews(train["review"][i]))
    
    # print an update ever 1000 reviews
    if((i+1) % 1000 == 0):
        print("Review %d" % (i+1))
    


'''
Model Preparation and Building
'''

# Initialize the CountVectorizer object for bag of words
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)

# Learn and Transform the training set into feature vectors
training_data_features = vectorizer.fit_transform(clean_train_reviews)

# convert to numpy array
training_data_features = training_data_features.toarray()

print(training_data_features.shape) # 25000 rows, 5000 features (one for each word)

# Look at the vocabulary
vocabulary = vectorizer.get_feature_names()
print(vocabulary)
# Sum the counts of each vocab word
dist = np.sum(training_data_features, axis = 0)

# print vocab word and number of times it appears
for tag, count in zip(vocabulary, dist):
    print(count, tag)


# Random Forest
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

# fit the model
forest = forest.fit(training_data_features, train["sentiment"])
print(forest)


'''
Making Predictions
'''


# append test reviews to empty list
num_reviews = len(test["review"])
clean_test_reviews = []

# clean the test set
for i in xrange(0, num_reviews):
    # print an update ever 1000 reviews
    if((i+1) % 1000 == 0):
        print("Review %d" % (i+1))
    
    clean_review = clean_reviews(test["review"][i])
    clean_test_reviews.append(clean_review)

# bag of words for test set (built in way)
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# predict on the test data
rf_predictions = forest.predict(test_data_features)
































































