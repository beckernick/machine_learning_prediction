# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 13:16:09 2015

@author: nickbecker
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort(['sqft_living','price'])


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


poly1_data = polynomial_dataframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price']


## Regression using Sci-kit learn

# fit a degree-1 model
model_1 = linear_model.LinearRegression()
model_1.fit(poly1_data['power_1'].reshape((len(poly1_data), 1)),
            poly1_data['price'])
model_1.coef_
model_1.intercept_
predictions_1 = model_1.predict(poly1_data['power_1'].reshape((len(poly1_data), 1)))

# plot the data with the model's predicted values
plt.plot(poly1_data['power_1'],poly1_data['price'], '.',
poly1_data['power_1'], predictions_1, '-')


# fit a degree-2 model
poly2_data = polynomial_dataframe(sales['sqft_living'], 2)
poly2_data['price'] = sales['price']
features = ['power_1', 'power_2']

model_2 = linear_model.LinearRegression()
model_2.fit(poly2_data[features],
            poly2_data['price'])
model_2.coef_
model_2.intercept_
predictions_2 = model_2.predict(poly2_data[features])

# plot the data with the model's predicted values
plt.plot(poly2_data['power_1'],poly2_data['price'], '.',
poly2_data['power_1'], predictions_2, '-')



# fit a degree-15 model
poly15_data = polynomial_dataframe(sales['sqft_living'], 15)
poly15_data['price'] = sales['price']
features = poly15_data.columns[:-1]

model_15 = linear_model.LinearRegression()
model_15.fit(poly15_data[features],
            poly15_data['price'])
model_15.coef_
model_15.intercept_
predictions_15 = model_15.predict(poly15_data[features])

# plot the data with the model's predicted values
plt.plot(poly15_data['power_1'],poly15_data['price'], '.',
poly15_data['power_1'], predictions_15, '-')





### Fitting 15th degree polynomimal on 4 datasets
kc_house_set_1 = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
kc_house_set_2 = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
kc_house_set_3 = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
kc_house_set_4 = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)



# Set 1
set1_poly15_data = polynomial_dataframe(kc_house_set_1['sqft_living'], 15)
set1_poly15_data['price'] = sales['price']
features = set1_poly15_data.columns[:-1]

set1_model_15 = linear_model.LinearRegression()
set1_model_15.fit(set1_poly15_data[features],
            set1_poly15_data['price'])
set1_model_15.coef_
set1_model_15.intercept_
set1_predictions_15 = set1_model_15.predict(set1_poly15_data[features])

# plot the data with the model's predicted values
plt.plot(set1_poly15_data['power_1'],set1_poly15_data['price'], '.',
set1_poly15_data['power_1'], set1_predictions_15, '-')




# Set 2
set2_poly15_data = polynomial_dataframe(kc_house_set_2['sqft_living'], 15)
set2_poly15_data['price'] = kc_house_set_2['price']
features = set2_poly15_data.columns[:-1]

set2_model_15 = linear_model.LinearRegression()
set2_model_15.fit(set2_poly15_data[features],
            set2_poly15_data['price'])
set2_model_15.coef_
set2_model_15.intercept_
set2_predictions_15 = set2_model_15.predict(set2_poly15_data[features])

# plot the data with the model's predicted values
plt.plot(set2_poly15_data['power_1'],set2_poly15_data['price'], '.',
set2_poly15_data['power_1'], set2_predictions_15, '-')




# Set 3
set3_poly15_data = polynomial_dataframe(kc_house_set_3['sqft_living'], 15)
set3_poly15_data['price'] = kc_house_set_3['price']
features = set3_poly15_data.columns[:-1]

set3_model_15 = linear_model.LinearRegression()
set3_model_15.fit(set3_poly15_data[features],
            set3_poly15_data['price'])
set3_model_15.coef_
set3_model_15.intercept_
set3_predictions_15 = set3_model_15.predict(set3_poly15_data[features])

# plot the data with the model's predicted values
plt.plot(set3_poly15_data['power_1'],set3_poly15_data['price'], '.',
set3_poly15_data['power_1'], set3_predictions_15, '-')



# Set 4
set4_poly15_data = polynomial_dataframe(kc_house_set_4['sqft_living'], 15)
set4_poly15_data['price'] = kc_house_set_4['price']
features = set4_poly15_data.columns[:-1]

set4_model_15 = linear_model.LinearRegression()
set4_model_15.fit(set4_poly15_data[features],
            set4_poly15_data['price'])
set4_model_15.coef_
set4_model_15.intercept_
set4_predictions_15 = set4_model_15.predict(set4_poly15_data[features])

# plot the data with the model's predicted values
plt.plot(set4_poly15_data['power_1'],set4_poly15_data['price'], '.',
set4_poly15_data['power_1'], set4_predictions_15, '-')




### 

kc_house_train = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_train_data.csv', dtype=dtype_dict)
kc_house_validation = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_valid_data.csv', dtype=dtype_dict)
kc_house_test = pd.read_csv('/Users/nickbecker/Documents/Github/machine_learning_prediction/housing_prices_prediction/data/wk3_kc_house_test_data.csv', dtype=dtype_dict)

best_rss = 1e20 # initialize to a high value
for degree in range(1,16):
    poly_data_temp = polynomial_dataframe(kc_house_train['sqft_living'], degree)
    poly_data_temp['price'] = kc_house_train['price']
    
    features_temp = poly_data_temp.columns[:-1]
        
    model_temp = linear_model.LinearRegression()
    model_temp.fit(poly_data_temp[features_temp], poly_data_temp['price'])
    
    validation_temp = polynomial_dataframe(kc_house_validation['sqft_living'], degree)
    validation_temp['price'] = kc_house_validation['price']
    validation_predictions = model_temp.predict(validation_temp[features_temp])
        
    # compute validation RSS
    model_temp_ssr = sum((validation_predictions - validation_temp['price'])**2)
    
    if model_temp_ssr < best_rss:
        best_rss = model_temp_ssr    
        best_model = "Degree: " + str(degree)
        
print(best_model)
    


# degree 6 on test data
train_6 = polynomial_dataframe(kc_house_train['sqft_living'], 6)
train_6['price'] = kc_house_train['price']
features_6 = train_6.columns[:-1]

model_6 = linear_model.LinearRegression()
model_6.fit(train_6[features_6], train_6['price'])
    
test_6 = polynomial_dataframe(kc_house_test['sqft_living'], 6)
test_6['price'] = kc_house_test['price']

test_6_predictions = model_6.predict(test_6[features_6])
    
# compute test 4 RSS
test_6_rss = sum((test_6_predictions - test_6['price'])**2)
print(test_6_rss)













































