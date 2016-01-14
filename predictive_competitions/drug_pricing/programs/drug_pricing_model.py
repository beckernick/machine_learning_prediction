# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 21:27:39 2016

@author: nickbecker
"""
import os
import pandas as pd
import numpy as np


os.chdir('/Users/nickbecker/Documents/Github/machine_learning_prediction/predictive_competitions/drug_pricing/')

train_raw = pd.read_csv('data/CAX_Bidding_TRAIN_Molecule_3_4_5.csv')
test_post_raw = pd.read_csv('data/CAX_Bidding_TEST_Molecule_6_Post_LOE.csv')

test_post_raw = test_post_raw.drop('Winning_price_per_standard_unit', axis = 1)


test_post_raw.head()
test_post_raw.columns
































