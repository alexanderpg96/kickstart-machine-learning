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
from tensorflow.contrib.graph_editor.select import compute_boundary_ts
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# load data and divide it into two subsets, used for training and testing
# category, time between deadline and launch, goal, amount pledged, number of backers
# category: 3, goal: 6, pledged: 8, backers: 10, time between deadline and launch: 15
kickstartData = download_data('Choppeddatasetv4.csv', [3, 6, 8, 10, 15]).values 
kickstartLabels = download_data('Choppeddatasetv4.csv',[9]).values
labels, kickstartLabels = np.unique(kickstartLabels[:,0], return_inverse=True)
columns = ['category','goal','pledged','backers','days_between']
# list of labels
print("The labels are",labels[0],": 0 and", labels[1], ": 1.")

df = pd.DataFrame(
    data=kickstartData,
    columns=columns)

#kickstartLabels = download_data('labels.csv',[0])


### display Success and failures by category ###
pd.crosstab(kickstartData[:,0],kickstartLabels).plot(kind='bar')
plt.title('Success and Failures by Category')
plt.xlabel('Job')
plt.ylabel('Frequency of Genres')
plt.savefig('Genres')
plt.show()

# replace category string values to floats
category, category_num = np.unique(kickstartData[:,0], return_inverse=True)
kickstartData[:,0] = category_num

# list of categories
print("List of categories:",category)


print("Data Shape" , kickstartData.shape)
print("Label Data Shape", kickstartLabels.shape)

# split the data into testing and training with a 20/80 split
x_train, x_test, y_train, y_test = train_test_split(kickstartData, kickstartLabels, test_size=0.20)

df = pd.DataFrame(
    data=x_test,
    columns=columns)
goal_max = df['goal'].max()
goal_min = df['goal'].min()

pt = x_test[:,1].transpose()
g_min = x_test[:,1].min()
g_max = x_test[:,1].max()


filter_val = np.array([20000000.0,40000000.0,60000000.0,80000000.0,100000000.0])
filter_val = filter_val.astype(float)
np.dtype()
categ = np.digitize(pt,filter_val,right=True)

print(g_min, g_max)
filter_values = [g_min+5000,g_min+10000,g_max]
p_u, pt = np.unique(pt, return_inverse = True)
print(p_u)
p = pd.cut(pt, 5).value_counts()
print(p)
# print(p)
# pd_group_cut = df.groupby(pd.cut(x_test[:,0], 15)).sum()
# print(pd_group_cut)

# df['goal'].hist(normed=1, bins=20, stacked=False, alpha=.5)
# plt.hist(df['goal'], bins=100, range=(goal_min,goal_max))
# print(goal.hist(data, column=None, by=None, grid=True, xlabelsize=None,
#                xrot=None, ylabelsize=None, yrot=None, ax=None, sharex=False,
#                sharey=False, figsize=None, layout=None, bins=10, **kwds))
# plt.show()

# sklearn Logistic Regression
# logisticRegr = LogisticRegression()

# reshape y_train to (39999,)
# y_train = np.array(y_train)
# y_train = y_train.reshape(39999,)
# train class instance
# logisticRegr.fit(x_train, y_train)

# predictions
# predictions = logisticRegr.predict(x_test)
# print(predictions)
# accuracy
# accuracy = logisticRegr.score(x_test, y_test)


# visualize data using functions in the library pylab
# Y = np.array(kickstartLabels)
# Y = Y.astype(float)
# Y = Y.reshape(49999,)
# print(Y.shape)
# 
# X = np.array(kickstartData)
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
# X = min_max_scaler.fit_transform(X)
# 
# # number of fails and success
# sns.countplot(x=Y, palette='hls')
# xlabel('Failed 0.0 : Successful 1.0')
# plt.savefig('Failed_Successful_Plot')
# 
# plt.plot(y_test,predictions, '.')
# x = np.linspace(0,330,100)
# y = x
# plt.plot(x,y)
# plt.show()
# 
# CM, acc, arrR, arrP = func_confusion_matrix(y_test, predictions)
# print("Confusion matrix",CM)
# print("Accuracy: ", acc)
# print("Per-class precision rate: ",arrP)
# print("Per-class recall rate: ",arrR)
