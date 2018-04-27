from download_data import download_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from conf_matrix import func_confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from numpy import array, where
from pylab import scatter, show, legend, xlabel, ylabel

import seaborn as sns
from seaborn import palettes
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# load data and divide it into two subsets, used for training and testing
# category, time between deadline and launch, goal, amount pledged, number of backers
# category: 3, goal: 6, pledged: 8, state: 9, backers: 10, time between deadline and launch: 15
kickstartData = download_data('Choppeddatasetv4.csv', [3, 6, 8, 10, 15]).values 
kickstartLabels = download_data('labels.csv',[0]) 

# replace category string values to floats
#[ 'Art', 'Comics', 'Crafts', 'Dance', 'Design', 'Fashion', 'Film & Video', 'Food',
# 'Games', 'Journalism', 'Music', 'Photography', 'Publishing', 'Technology', 'Theater']
category, category_num = np.unique(kickstartData[:,0], return_inverse=True)
kickstartData[:,0] = category_num

print("Data Shape" , kickstartData.shape)
print("Label Data Shape", kickstartLabels.shape)

# split the data into testing and training with a 20/80 split
x_train, x_test, y_train, y_test = train_test_split(kickstartData, kickstartLabels, test_size=0.20)

# sklearn Logistic Regression
logisticRegr = LogisticRegression()

# reshape y_train to (39999,)
y_train = np.array(y_train)
y_train = y_train.reshape(39999,)
# train class instance
logisticRegr.fit(x_train, y_train)

# predictions
predictions = logisticRegr.predict(x_test)
print(predictions)
# accuracy
accuracy = logisticRegr.score(x_test, y_test)


# visualize data using functions in the library pylab
Y = np.array(kickstartLabels)
Y = Y.astype(float)
Y = Y.reshape(49999,)
print(Y.shape)

X = np.array(kickstartData)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
X = min_max_scaler.fit_transform(X)

# visualize data using functions in the library pylab 
pos = where(Y == 1)
neg = where(Y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Feature 1: score 1')
ylabel('Feature 2: score 2')
legend(['Label:  Admitted', 'Label: Not Admitted'])
show()

# number of fails and success
sns.countplot(x=Y, palette='hls')
xlabel('Failed 0.0 : Successful 1.0')
plt.savefig('Failed_Successful_Plot')


pd.crosstab(kickstartData[:,0],kickstartLabels).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')

plt.show()


CM, acc, arrR, arrP = func_confusion_matrix(y_test.values[:,0], predictions)
print("Confusion matrix",CM)
print("Accuracy: ", acc)
print("Per-class precision rate: ",arrP)
print("Per-class recall rate: ",arrR)
