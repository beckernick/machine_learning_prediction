# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 22:49:02 2016

@author: nickbecker
"""

import pandas as pd
import numpy as np

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':int, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/kc_house_data.csv', dtype=dtype_dict)
testing = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_train_data.csv', dtype=dtype_dict)



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





































































