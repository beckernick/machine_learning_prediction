# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 21:27:50 2016

@author: nickbecker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales = pd.read_csv("/users/nickbecker/Python_Projects/king_county_house_sales/kc_house_data.csv",
                    dtype = dtype_dict)
sales = sales.sort(['sqft_living','price'])

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

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant=False):
    # function to compute the partial derivative using matrix algebra
    if feature_is_constant == True:
        derivative = 2*np.dot(errors, feature)
    else:
        derivative = 2*np.dot(errors, feature) + 2*l2_penalty*weight
    return derivative


(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
my_weights = np.array([1., 10.])
test_predictions = predict_output(example_features, my_weights) 
errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False)
print np.sum(errors*example_features[:,1])*2+20.
print ''

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True)
print np.sum(errors)*2.

def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    # Function to perform ridge regression gradient descent
    
    weights = np.array(initial_weights) # Make sure it's a numpy array
    iteration = 0
    
    while iteration <= max_iterations: # Only run a specified number of times
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output  
        
        for i in xrange(len(weights)): # loop over each weight
            # At i = 0, computing for the constant (so Feature_is_constant = True)
            derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, i == 0)
            weights[i] = weights[i] - step_size*derivative
            iteration = iteration + 1
        #print(weights)
    return weights


# Read in in training and test data
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
train = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_train_data.csv', dtype=dtype_dict)
test = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_test_data.csv', dtype=dtype_dict)

# Create simple algorithm testing example
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test, simple_features, my_output)

# L2 penalty = 0 regression weights
initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000

simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights = initial_weights,
                                                             step_size = step_size, l2_penalty = 0, max_iterations = 1000)
print(simple_weights_0_penalty)

# L2 penalty = 1e11 regression weights
simple_weights_1e11_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights = initial_weights,
                                                             step_size = step_size, l2_penalty = 1e11, max_iterations = 1000)
print(simple_weights_1e11_penalty)

# Plot the models
plt.plot(simple_feature_matrix,output,'k.',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_1e11_penalty),'r-')




### RSS on Test data for all zeros, l2 = 0 weights, l2 = 1e11 weights
preds_all_zero = predict_output(simple_test_feature_matrix, np.array([0.,0.]))
preds_0_penalty_weights = predict_output(simple_test_feature_matrix, simple_weights_0_penalty)
preds_1e11_penalty_weights = predict_output(simple_test_feature_matrix, simple_weights_1e11_penalty)

RSS_all_zero = sum((preds_all_zero - test_output)**2)
RSS_0_penalty = sum((preds_0_penalty_weights - test_output)**2)
RSS_1e11_penatly = sum((preds_1e11_penalty_weights - test_output)**2)

print(RSS_all_zero)
print(RSS_0_penalty)
print(RSS_1e11_penatly)




#### Model with 2 features
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test, model_features, my_output)

initial_weights = np.array([0.0, 0.0, 0.0])
step_size = 1e-12
max_iterations=1000

multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights = initial_weights,
                                                             step_size = step_size, l2_penalty = 0, max_iterations = 1000)
print(multiple_weights_0_penalty)

multiple_weights_1e11_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights = initial_weights,
                                                             step_size = step_size, l2_penalty = 1e11, max_iterations = 1000)
print(multiple_weights_1e11_penalty)

preds_all_zero = predict_output(test_feature_matrix, np.array([0.0,0.0,0.0]))
preds_0_penalty_weights = predict_output(test_feature_matrix, multiple_weights_0_penalty)
preds_1e11_penalty_weights = predict_output(test_feature_matrix, multiple_weights_1e11_penalty)

RSS_all_zero = sum((preds_all_zero - test_output)**2)
RSS_0_penalty = sum((preds_0_penalty_weights - test_output)**2)
RSS_1e11_penatly = sum((preds_1e11_penalty_weights - test_output)**2)

print(RSS_all_zero)
print(RSS_0_penalty)
print(RSS_1e11_penatly)


test_output[0] - preds_0_penalty_weights[0]
test_output[0] - preds_1e11_penalty_weights[0]












########### SKLEARN RIDGE #################

# Create simple algorithm testing example
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test, simple_features, my_output)

small_pen = 0
large_pen = 1e11

model = linear_model.Ridge(alpha=small_pen, normalize=False)
model.fit(simple_feature_matrix, output)
model.coef_

model_preds = model.predict(simple_test_feature_matrix)
model_preds[1:5]
RSS_model = sum((model_preds - test_output)**2)
print(RSS_model)

## 2 feature model

#### Model with 2 features
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test, model_features, my_output)

model2 = linear_model.Ridge(alpha=small_pen, normalize=False)
model2.fit(feature_matrix, output)
model2.coef_

model2_preds = model2.predict(test_feature_matrix)
RSS_model2 = sum((model2_preds - test_output)**2)
print(RSS_model2)












