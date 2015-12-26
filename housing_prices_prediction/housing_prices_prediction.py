# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 10:24:43 2015

@author: nickbecker
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
train = pd.read_csv("/users/nickbecker/Python_Projects/king_county_house_sales/kc_house_train_data.csv",
                    dtype = dtype_dict)
test = pd.read_csv("/users/nickbecker/Python_Projects/king_county_house_sales/kc_house_test_data.csv",
                   dtype = dtype_dict)


### Feature engineering
train['bedrooms_squared'] = train.bedrooms*train.bedrooms
train['bed_bath_rooms'] = train.bedrooms*train.bathrooms
train['log_sqft_living'] = np.log(train.sqft_living)
train['lat_plus_long'] = train.lat + train.long

test['bedrooms_squared'] = test.bedrooms*test.bedrooms
test['bed_bath_rooms'] = test.bedrooms*test.bathrooms
test['log_sqft_living'] = np.log(test.sqft_living)
test['lat_plus_long'] = test.lat + test.long


test.bedrooms_squared.describe()
test.bed_bath_rooms.describe()
test.log_sqft_living.describe()
test.lat_plus_long.describe()


### Linear Regression

## Using Statsmodels

# model 1
features_1 = sm.add_constant(train[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']])
target_1 = train['price']

model_1 = sm.OLS(target_1, features_1).fit()
model_1.params
model_1.ssr
model_1.summary()

# model 2
features_2 = sm.add_constant(train[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']])
target_2 = train['price']

model_2 = sm.OLS(target_2, features_2).fit()
model_2.params
model_2.ssr
model_2.summary()

# model 3
features_3 = sm.add_constant(train[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long',
                                    'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']])
target_3 = train['price']

model_3 = sm.OLS(target_3, features_3).fit()
model_3.params
model_3.ssr
model_3.summary()


## Using Sci-kit learn
from sklearn import linear_model

# model 1
features_1 = train[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']]
target_1 = train['price']

sk_model_1 = linear_model.LinearRegression()
sk_model_1.fit(features_1, target_1)
sk_model_1.coef_
sk_model_1.intercept_

# model 2
features_2 = train[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']]
target_2 = train['price']

sk_model_2 = linear_model.LinearRegression()
sk_model_2.fit(features_2, target_2)
sk_model_2.coef_
sk_model_2.intercept_

# model 3
features_3 = train[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long',
                                    'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']]
target_3 = train['price']

sk_model_3 = linear_model.LinearRegression()
sk_model_3.fit(features_3, target_3)
sk_model_3.coef_
sk_model_3.intercept_

























































