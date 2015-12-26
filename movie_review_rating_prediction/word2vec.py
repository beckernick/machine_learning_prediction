# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 19:07:09 2015

@author: nickbecker
"""

'''
word2vec in python
'''

import pandas as pd
import numpy as np

# read in the data
train = pd.read_csv("/users/nickbecker/Python_Projects/imdb_data/labeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3, encoding = 'utf-8')
test = pd.read_csv("/users/nickbecker/Python_Projects/imdb_data/testData.tsv",
                    header=0, delimiter="\t", quoting=3, encoding = 'utf-8')
unlabeled_train = pd.read_csv("/users/nickbecker/Python_Projects/imdb_data/unlabeledTrainData.tsv", header = 0,
                              delimiter = "\t", quoting=3, encoding = 'utf-8')
                              
train.shape
train.columns.values
train.head()
print(train["review"][0])


# Functions to clean the data
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist(review, remove_stopwords=False):
    # convert document to sequence of words
    # returns list of wirds
    
    # remove HTML    tags 
    review_text = BeautifulSoup(review).get_text()
    
    # remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    
    # convert to lower case and then split into list of words
    words = review_text.lower().split()
    
    # remove stopwords (if active)
    if remove_stopwords == True:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    
    # return list of words
    return(words)


'''
word2vec algorithm needs single sentences
Use NLTK's punctuation tokenizer to split paragraphs into sentences
'''

import nltk.data
#nltk.download()

# load punkt tokenizer
tokenizer = nltk.data.load('/users/nickbecker/nltk_data/tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # Function to split review into sentences
    # Return list of sentences

    # Use NLTK tokenizer to split the paragraph into sentences    
    raw_sentences = tokenizer.tokenize(review.strip())
    
    # loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # if a sentence is not empty, get a list of words using the function above
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
        
    
    # return list of sentences (list of lists, since each sentence is a list of words)
    return sentences


sentences = [] # initialize empty list
print("Parsing sentences from training set")
for review in train['review']:
    sentences += review_to_sentences(review, tokenizer)

print(len(sentences))
print(sentences[0])
print(sentences[1])



#### Training the model ####
import logging 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# set values for word2vec parameters
num_features = 300   # word vector dimensionality
min_word_count = 40     # minimum word count
number_workers = 4      # number of threads to run in parallel
context = 10            # context window size
downsampling = .001     # downsample setting for frequent words

# initialize and train the model
from gensim.models import word2vec

print("Training model...")
model = word2vec.Word2Vec(sentences, workers=number_workers, size=num_features, min_count = min_word_count,
                          window = context, sample = downsampling)

model.init_sims(replace = True)

# save the model
model_name = "300features_40minwords_10context"
model.save('/users/nickbecker/documents/Github/machine_learning_prediction/movie_review_rating_prediction/' + model_name)



##### Looking at the model ####

# Which word doesn't match?
model.doesnt_match("man woman child kitchen".split())
model.doesnt_match("france england germany berlin".split()) # even subtle differences
model.doesnt_match("paris berlin london austria".split()) # not perfect though

# Most similar words
model.most_similar("man")
model.most_similar("queen")
model.most_similar("awful")

# access individual word vectors
model["flower"]




###### Vector Averaging for Prediction ######

def makeFeatureVec(words, model, num_features):
    # average all word vectors in a paragraph
    # pre-initialize empty array for speed
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    
    index2word_set = set(model.index2word)
    
    # loop over each word in the review and if its in the model's vocabulary
    # add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])
    
    # divid result by number of words to get average
    featureVec_avg = np.divide(featureVec, nwords)
    return(featureVec_avg)


def getAvgFeatureVecs(reviews, model, num_features):
    # given set of reviews (each one a list of words), calculate
    # average feature vector for each one and return a 2d numpy array
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype = "float32")
    
    # loop through reviews
    for review in reviews:
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))
        
        # call above function that makes avg feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        
        counter += 1
    return(reviewFeatureVecs)
            


# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Remove stopwords.

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print("Creating average feature vecs for test reviews")

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)




##### Train a random forest classifier ######
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100) # 100 trees
print("Fitting random forest to training data")
forest = forest.fit(trainDataVecs, train["sentiment"])

predictions = forest.predict(testDataVecs)

output = pd.DataFrame(data = {"id":test["id"], "sentiment":predictions})
output.to_csv("/users/nickbecker/documents/Github/machine_learning_prediction/movie_review_rating_prediction/Word2Vec_AvgVectors.csv", index = False, quoting = 3)


















































