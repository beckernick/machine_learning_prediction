# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 22:09:01 2015

@author: nickbecker
"""

import pandas as pd
import numpy as np
from math import sqrt

def polynomial_dataframe(feature, degree):
    # Returns a data frame of a Series raised to powers up to degree
    poly_dataframe = pd.DataFrame()
    poly_dataframe['power_1'] = feature

    if degree > 1:
        # loop over the remaining degrees:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_dataframe[name] = poly_dataframe['power_1'].apply(lambda x: x**power)
    
    return poly_dataframe



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



# Create predictions vector
def predict_output(feature_matrix, weights):
    # feature matrix is a numpy matrix of features as columns; weights is a corresponding numpy array
    # create the predictions vector using matrix algebra
    predictions = np.dot(feature_matrix, weights)
    return(predictions)


# function to compute the partial derivative using matrix algebra
def feature_derivative(errors, feature):
    derivative = 2*np.dot(errors, feature)
    return(derivative)


# Gradient Descent
def gradient_descent_multiple_regression(feature_matrix, output, initial_weights, step_size, tolerance):
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
        




