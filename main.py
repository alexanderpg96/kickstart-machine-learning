from download_data import download_data
import numpy as np
import matplotlib.pyplot as plt
from data_norm import rescaleMatrix

# parameters 
ALPHA = 0.1
MAX_ITER = 500

# load data and divide it into two subsets, used for training and testing
# category, time between deadline and launch, goal, amount pledged, number of backers
# category - 2, goal - 6, pledged - 8, state - 9, backers - 10

kickstart = download_data('Choppeddatasetv3.csv', [6, 8, 10]).values 

# Normalize data
kickstart = rescaleMatrix(kickstart) 
  
# split the data 80/20 for training and testing
# 50,000 rows total 
# training data
kickTrain = kickstart[0:40000, :]
# testing data
kickTest = kickstart[40000:len(kickstart),:]

print(len(kickTrain))
print(len(kickTest))