from download_data import download_data
import numpy as np
import matplotlib.pyplot as plt
from data_norm import rescaleMatrix
from conf_matrix import func_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from numpy import array




# load data and divide it into two subsets, used for training and testing
# category, time between deadline and launch, goal, amount pledged, number of backers
# category - 2, goal - 6, pledged - 8, state - 9, backers - 10, time between deadline and launch -15

kickstartData = download_data('Choppeddatasetv4.csv', [2, 6, 8, 10, 15]).values 
kickstartLabels = download_data('labels.csv',[0]) 

print("Data Shape" , kickstartData.shape)
print("Label Data Shape", kickstartLabels.shape)

# split the data into testing and training with a 20/80 split
x_train, x_test, y_train, y_test = train_test_split(kickstartData, kickstartLabels, test_size=0.20, random_state=42)


logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)
accuracy = logisticRegr.score(x_test, y_test)

CM, acc, arrR, arrP = func_confusion_matrix(y_test.values[:,0], predictions)
print("Confusion matrix",CM)
print("Accuracy: ", acc)
print("Per-class precision rate: ",arrP)
print("Per-class recall rate: ",arrR)






