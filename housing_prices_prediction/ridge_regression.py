# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 22:15:41 2015

@author: nickbecker
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from math import sqrt
import os
os.chdir('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/utility_functions')
import algorithm_utility_functions as alg
os.chdir('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/')
import matplotlib.pyplot as plt

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales = pd.read_csv("/users/nickbecker/Python_Projects/king_county_house_sales/kc_house_data.csv",
                    dtype = dtype_dict)
sales = sales.sort(['sqft_living','price'])

l2_small_penalty = 1.5e-5

poly15_data = alg.polynomial_dataframe(sales['sqft_living'], 15) 

model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model.fit(poly15_data, sales['price'])
model.coef_


##### Ridge Regression 15th degree polynomial for the four datasets
kc_house_set_1 = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
kc_house_set_2 = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
kc_house_set_3 = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
kc_house_set_4 = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)



### Small L2 Penalty 
l2_small_penalty=1e-9

# Set 1
set1_poly15_data = alg.polynomial_dataframe(kc_house_set_1['sqft_living'], 15)
set1_poly15_data['price'] = kc_house_set_1['price']
features = set1_poly15_data.columns[:-1]

set1_ridge = linear_model.Ridge(alpha = l2_small_penalty, normalize = True)
set1_ridge.fit(set1_poly15_data[features],
            set1_poly15_data['price'])
set1_ridge.coef_
set1_ridge.intercept_
set1_ridge_predictions = set1_ridge.predict(set1_poly15_data[features])

# plot the data with the model's predicted values
plt.plot(set1_poly15_data['power_1'],set1_poly15_data['price'], '.',
set1_poly15_data['power_1'], set1_ridge_predictions, '-')



# Set 2
set2_poly15_data = alg.polynomial_dataframe(kc_house_set_2['sqft_living'], 15)
set2_poly15_data['price'] = kc_house_set_2['price']
features = set2_poly15_data.columns[:-1]

set2_ridge = linear_model.Ridge(alpha = l2_small_penalty, normalize = True)
set2_ridge.fit(set2_poly15_data[features],
            set2_poly15_data['price'])
set2_ridge.coef_
set2_ridge.intercept_
set2_ridge_predictions = set2_ridge.predict(set2_poly15_data[features])

# plot the data with the model's predicted values
plt.plot(set2_poly15_data['power_1'],set2_poly15_data['price'], '.',
set2_poly15_data['power_1'], set2_ridge_predictions, '-')


# Set 3
set3_poly15_data = alg.polynomial_dataframe(kc_house_set_3['sqft_living'], 15)
set3_poly15_data['price'] = kc_house_set_3['price']
features = set3_poly15_data.columns[:-1]

set3_ridge = linear_model.Ridge(alpha = l2_small_penalty, normalize = True)
set3_ridge.fit(set3_poly15_data[features],
            set3_poly15_data['price'])
set3_ridge.coef_
set3_ridge.intercept_
set3_ridge_predictions = set3_ridge.predict(set3_poly15_data[features])

# plot the data with the model's predicted values
plt.plot(set3_poly15_data['power_1'],set3_poly15_data['price'], '.',
set3_poly15_data['power_1'], set3_ridge_predictions, '-')




# Set 4
set4_poly15_data = alg.polynomial_dataframe(kc_house_set_4['sqft_living'], 15)
set4_poly15_data['price'] = kc_house_set_4['price']
features = set4_poly15_data.columns[:-1]

set4_ridge = linear_model.Ridge(alpha = l2_small_penalty, normalize = True)
set4_ridge.fit(set4_poly15_data[features],
            set4_poly15_data['price'])
set4_ridge.coef_
set4_ridge.intercept_
set4_ridge_predictions = set4_ridge.predict(set4_poly15_data[features])

# plot the data with the model's predicted values
plt.plot(set4_poly15_data['power_1'],set4_poly15_data['price'], '.',
set4_poly15_data['power_1'], set4_ridge_predictions, '-')



### Large L2 Penalty
l2_large_penalty=1.23e2

# Set 1
set1_poly15_data = alg.polynomial_dataframe(kc_house_set_1['sqft_living'], 15)
set1_poly15_data['price'] = kc_house_set_1['price']
features = set1_poly15_data.columns[:-1]

set1_ridge = linear_model.Ridge(alpha = l2_large_penalty, normalize = True)
set1_ridge.fit(set1_poly15_data[features],
            set1_poly15_data['price'])
set1_ridge.coef_
set1_ridge.intercept_
set1_ridge_predictions = set1_ridge.predict(set1_poly15_data[features])

# plot the data with the model's predicted values
plt.plot(set1_poly15_data['power_1'],set1_poly15_data['price'], '.',
set1_poly15_data['power_1'], set1_ridge_predictions, '-')


# Set 2
set2_poly15_data = alg.polynomial_dataframe(kc_house_set_2['sqft_living'], 15)
set2_poly15_data['price'] = kc_house_set_2['price']
features = set2_poly15_data.columns[:-1]

set2_ridge = linear_model.Ridge(alpha = l2_large_penalty, normalize = True)
set2_ridge.fit(set2_poly15_data[features],
            set2_poly15_data['price'])
set2_ridge.coef_
set2_ridge.intercept_
set2_ridge_predictions = set2_ridge.predict(set2_poly15_data[features])

# plot the data with the model's predicted values
plt.plot(set2_poly15_data['power_1'],set2_poly15_data['price'], '.',
set2_poly15_data['power_1'], set2_ridge_predictions, '-')




# Set 3
set3_poly15_data = alg.polynomial_dataframe(kc_house_set_3['sqft_living'], 15)
set3_poly15_data['price'] = kc_house_set_3['price']
features = set3_poly15_data.columns[:-1]

set3_ridge = linear_model.Ridge(alpha = l2_large_penalty, normalize = True)
set3_ridge.fit(set3_poly15_data[features],
            set3_poly15_data['price'])
set3_ridge.coef_
set3_ridge.intercept_
set3_ridge_predictions = set3_ridge.predict(set3_poly15_data[features])

# plot the data with the model's predicted values
plt.plot(set3_poly15_data['power_1'],set3_poly15_data['price'], '.',
set3_poly15_data['power_1'], set3_ridge_predictions, '-')



# Set 4
set4_poly15_data = alg.polynomial_dataframe(kc_house_set_4['sqft_living'], 15)
set4_poly15_data['price'] = kc_house_set_4['price']
features = set4_poly15_data.columns[:-1]

set4_ridge = linear_model.Ridge(alpha = l2_large_penalty, normalize = True)
set4_ridge.fit(set4_poly15_data[features],
            set4_poly15_data['price'])
set4_ridge.coef_
set4_ridge.intercept_
set4_ridge_predictions = set4_ridge.predict(set4_poly15_data[features])

# plot the data with the model's predicted values
plt.plot(set4_poly15_data['power_1'],set4_poly15_data['price'], '.',
set4_poly15_data['power_1'], set4_ridge_predictions, '-')




###### K-Fold Cross Validation to select best L2 paramater
train_shuffled = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_test_data.csv', dtype=dtype_dict)

n = len(train_shuffled)
k = 10 # 10-fold cross-validation

train_shuffled[0:10]

for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print i, (start, end)

validation4 = train_shuffled[5818:7758]
print(int(round(validation4['price'].mean(), 0)))

def k_fold_cross_validation(k, l2_penalty, data, output):
    n = len(data)
    sum_validation_error = []
    for i in xrange(k):    
        start = (n*i)/k
        end = (n*(i+1))/k-1
        
        validation_features = data[start:end+1]
        validation_output = output[start:end+1]
        training_features = data[0:start].append(data[end+1:n])
        training_output = output[0:start].append(output[end+1:n])
        
        ridge_model = linear_model.Ridge(alpha = l2_penalty, normalize = True)
        ridge_model.fit(training_features, training_output)
        
        validation_predictions = ridge_model.predict(validation_features)
        
        # compute validation RSS
        RSS = sum((validation_predictions - validation_output)**2)
        sum_validation_error.append(RSS)

    average_validation_error = np.mean(sum_validation_error)
    return average_validation_error



# Loop over different L2 parameters
poly15_data = alg.polynomial_dataframe(train_shuffled['sqft_living'], 15) 

l2_list = np.logspace(3, 9, num=13)
best_cv_error = 1e20

for l2 in l2_list:
    cv_error = k_fold_cross_validation(10, l2, poly15_data, train_shuffled['price'])
    if cv_error < best_cv_error:
        best_cv_error = cv_error    
        best_l2 = "L2: " + str(l2)

print(best_l2)

# train model with best l2
best_model_train = linear_model.Ridge(alpha = 1000, normalize = True)
best_model_train.fit(poly15_data,
            train_shuffled['price'])

# predict on test data
poly15_test = alg.polynomial_dataframe(test['sqft_living'], 15) 

best_model_test_preds = best_model_train.predict(poly15_test)

# test RSS
print(sum((best_model_test_preds - test['price'])**2))





































