# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 20:31:51 2016

@author: nickbecker
"""

import pandas as pd
import numpy as np

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/kc_house_data.csv', dtype=dtype_dict)


from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

from sklearn import linear_model

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = linear_model.Lasso(alpha = 5e2, normalize = True)
model_all.fit(sales[all_features], sales['price'])
model_all.coef_
print(zip(all_features, model_all.coef_))


# Determine best L1 penalty by using a validation set
testing = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_valid_data.csv', dtype=dtype_dict)

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']


l1_penalties = np.logspace(1, 7, num=13)

for l1 in l1_penalties:
    #print(l1)
    training_model = linear_model.Lasso(alpha = l1, normalize = True)
    training_model.fit(training[all_features], training['price'])
    
    # compute RSS on validation
    validation_preds = training_model.predict(validation[all_features])
    RSS_validation = sum((validation_preds - validation['price'])**2)
    print(l1)   
    print(RSS_validation)
    print('\n')

# lowest RSS with l1 = 10
model_l1_10 = linear_model.Lasso(alpha = 10, normalize = True)
model_l1_10.fit(training[all_features], training['price'])
np.count_nonzero(model_l1_10.coef_) + np.count_nonzero(model_l1_10.intercept_) # 15 nonzero


# Limit a model to 7 features
l1_penalties_many = np.logspace(1,4, num = 20)
max_nonzeros = 7

nonzero_list = []
for l1 in l1_penalties_many:
    # fit a lasso model on training data
    training_model = linear_model.Lasso(alpha = l1, normalize = True)
    training_model.fit(training[all_features], training['price'])
    
    # extract weights and count number of nonzeros
    nonzero_number = np.count_nonzero(training_model.coef_) + np.count_nonzero(training_model.intercept_) # 15 nonzero
    nonzero_list.append(nonzero_number)

print(nonzero_list)
print(zip(l1_penalties_many, nonzero_list))

#More formally, find:
#The largest l1_penalty that has more non-zeros than max_nonzero (if we pick a penalty smaller than this value, we will definitely have too many non-zero weights)
#Store this value in the variable l1_penalty_min (we will use it later)
#The smallest l1_penalty that has fewer non-zeros than max_nonzero (if we pick a penalty larger than this value, we will definitely have too few non-zero weights)
#Store this value in the variable l1_penalty_max (we will use it later)
#Hint: there are many ways to do this, e.g.:
#Programmatically within the loop above
#Creating a list with the number of non-zeros for each value of l1_penalty and inspecting it to find the appropriate boundaries.

l1_penalty_min = 127.42749857031335
l1_penalty_max = 263.66508987303581

# exploring the narrower range of l1

l1_narrow_range = np.linspace(l1_penalty_min, l1_penalty_max, 20)

for l1 in l1_narrow_range:
    training_model = linear_model.Lasso(alpha = l1, normalize = True)
    training_model.fit(training[all_features], training['price'])
    
    # extract weights and count number of nonzeros
    nonzero_number = np.count_nonzero(training_model.coef_) + np.count_nonzero(training_model.intercept_) # 15 nonzero
    
    if nonzero_number == max_nonzeros:
        # compute RSS on validation
        validation_preds = training_model.predict(validation[all_features])
        RSS_validation = sum((validation_preds - validation['price'])**2)
        print(l1)   
        print(RSS_validation)
        print('\n')


# lowest 7 feature RSS with l1 = 163.279496282
model_best = linear_model.Lasso(alpha = 163.279496282, normalize = True)
model_best.fit(training[all_features], training['price'])
print(zip(all_features, model_best.coef_))

    
























