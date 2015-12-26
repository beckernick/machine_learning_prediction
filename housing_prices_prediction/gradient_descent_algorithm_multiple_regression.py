# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 14:48:42 2015

@author: nickbecker
"""

'''
Gradient Descent algorithm for multiple regression
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales = pd.read_csv("/users/nickbecker/Python_Projects/king_county_house_sales/kc_house_data.csv",
                    dtype = dtype_dict)
train = pd.read_csv("/users/nickbecker/Python_Projects/king_county_house_sales/kc_house_train_data.csv",
                    dtype = dtype_dict)
test = pd.read_csv("/users/nickbecker/Python_Projects/king_county_house_sales/kc_house_test_data.csv",
                   dtype = dtype_dict)

# Convert relevant data to numpy arrays

def get_numpy_data(pandas_df, features, output):
    # function to convert to numpy array with specified features and output
    pandas_df['constant'] = 1
    features = ['constant'] + features
    features_df = pandas_df[features]
    features_array = features_df.as_matrix()
    output_df = pandas_df[output]
    output_array = output_df.as_matrix()
    return(features_array, output_array)
    

(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')


# Create predictions vector

my_weights = np.array([1., 1.]) # the example weights


def predict_output(feature_matrix, weights):
    # feature matrix is a numpy matrix of features as columns; weights is a corresponding numpy array
    # create the predictions vector using matrix algebra
    predictions = np.dot(feature_matrix, weights)
    return(predictions)

example_predictions = predict_output(example_features, my_weights)
print(example_predictions[0])
print(example_predictions[1])


# function to compute the partial derivative using matrix algebra

def feature_derivative(errors, feature):
    derivative = 2*np.dot(errors, feature)
    return(derivative)

(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
my_weights = np.array([0., 0.])
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output
feature = example_features[:,0]
derivative = feature_derivative(errors, feature)
print(derivative)
print(-np.sum(example_output)*2)


# Gradient Descent
from math import sqrt

def gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute predictions
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output
        
        # initialize gradient sum squares as 0
        gradient_sum_squares = 0
        #print(weights)
        for i in range(len(weights)): # loop over each weight
            derivative = feature_derivative(errors, feature_matrix[:, i])
            gradient_sum_squares = gradient_sum_squares + derivative*derivative
            weights[i] = weights[i] - step_size*derivative
    
        # compute square root of gradient sum of squares to get graident magnitude
        gradient_magnitude = sqrt(gradient_sum_squares)
        #print(gradient_magnitude)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)
        

### Gradient Descent as Simple Linear Regression
simple_features = ['sqft_living']
my_output = 'price'
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

# training data
(simple_feature_matrix, output) = get_numpy_data(train, simple_features, my_output)

estimated_weights = gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)  
print(estimated_weights[1])

# predict on the test data
(test_simple_feature_matrix, test_output) = get_numpy_data(test, simple_features, my_output)
test_predictions = predict_output(test_simple_feature_matrix, estimated_weights)
print(test_predictions[0])

# compute test RSS
model1_ssr = sum((test_predictions - test['price'])**2)



### Multiple Regression with Gradient Descent
model_features = ['sqft_living', 'sqft_living15'] 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

model_weights = gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)  
print(model_weights)

# predict on test data
(test_feature_matrix, test_output) = get_numpy_data(test, model_features, my_output)
test_predictions_model2 = predict_output(test_feature_matrix, model_weights)
print(test_predictions_model2[0])

# compute test RSS
model2_ssr = sum((test_predictions_model2 - test['price'])**2)

print(abs(test.price[0] - test_predictions_model2[0]))
print(abs(test.price[0] - test_predictions[0]))

print(model1_ssr)
print(model2_ssr)












