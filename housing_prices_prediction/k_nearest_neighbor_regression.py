# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:01:33 2016

@author: nickbecker
"""

import numpy as np
import pandas as pd

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

training = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/kc_house_data_small_train.csv', dtype=dtype_dict)
test = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/kc_house_data_small_test.csv', dtype=dtype_dict)
validation = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/kc_house_data_validation.csv', dtype=dtype_dict)


def get_numpy_data(pandas_df, features, output):
    # function to convert to numpy array with specified features and output
    pandas_df['constant'] = 1
    features = ['constant'] + features
    features_df = pandas_df[features]
    features_array = features_df.as_matrix()
    output_df = pandas_df[output]
    output_array = output_df.as_matrix()
    return(features_array, output_array)


def predict_output(feature_matrix, weights):
    # feature matrix is a numpy matrix of features as columns; weights is a corresponding numpy array
    # create the predictions vector using matrix algebra
    predictions = np.dot(feature_matrix, weights)
    return(predictions)


def normalize_features(features):
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features / norms
    return (normalized_features, norms)


feature_list = ['bedrooms',  'bathrooms',  'sqft_living',  'sqft_lot',  
                'floors',   'waterfront',  'view',  'condition', 'grade',  
                'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',  
                'lat',  'long', 'sqft_living15',  'sqft_lot15']
                
features_train, output_train = get_numpy_data(training, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')

features_train, norms = normalize_features(features_train) # normalize training set features (columns)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms


# compute distance between two points
print(features_test[0])
print(features_train[9])

dist = np.sqrt(sum((features_test[0] - features_train[9])**2))
print(dist)

# compute distance between query and first 10 of training set
distances = []
for i in xrange(10):
    distances.append( np.sqrt(sum((features_test[0] - features_train[i])**2)) )

distances
print(zip(distances, range(10))) # index 8 house is closest (9th house)


# Vectorized simple distance computation
diff = features_train - features_test[0]
print diff[-1].sum() # sum of the feature differences between the query and last training house
# should print -0.0934339605842

# compute all distances
distances = np.sqrt( np.sum(diff**2, axis=1) )
print distances[100] # Euclidean distance between the query house and the 101th training house
# should print 0.0237082324496

def compute_distances(training_matrix, query_matrix):
    diff = training_matrix - query_matrix
    distances = np.sqrt( np.sum(diff**2, axis=1) )
    return(distances)

# query with features_test[2]
test2_dists = compute_distances(features_train, features_test[2])

print(np.argmin(test2_dists), np.amin(test2_dists))

# 1-NN predicted value for the query house
print(output_train[np.argmin(test2_dists)])



























































