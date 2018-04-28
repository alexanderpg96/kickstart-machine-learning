from download_data import download_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from conf_matrix import func_confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from numpy import array, where
from pylab import scatter, show, legend, xlabel, ylabel
from scipy import stats
from seaborn import palettes
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# load data and divide it into two subsets, used for training and testing
# category, time between deadline and launch, goal, amount pledged, number of backers
# category: 3, goal: 6, pledged: 8, backers: 10, time between deadline and launch: 15
kickstartData = download_data('Choppeddatasetv4.csv', [3, 6, 8, 10, 15]).values 
kickstartLabels = download_data('Choppeddatasetv4.csv',[9]).values
labels, kickstartLabels = np.unique(kickstartLabels[:,0], return_inverse=True)
columns = ['category','goal','pledged','backers','days_between']

### list of labels ###
print("The labels are",labels[0],": 0 and", labels[1], ": 1.")

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

### list of categories ###
print("List of categories:",category)

### data and label shapes ###
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

# accuracy
accuracy = logisticRegr.score(x_test, y_test)

### data for countplot ###
Y = kickstartLabels.astype(float)
Y = Y.reshape(49999,)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
X = min_max_scaler.fit_transform(kickstartData)
 
# number of fails and success
sns.countplot(x=Y, palette='hls')
xlabel("Failed 0.0 : Successful 1.0")
plt.show()
plt.savefig("Failed_Successful_Plot")
 
### get confusion matrix, accuracy, precision rate, and recall rate
CM, acc, arrR, arrP = func_confusion_matrix(y_test, predictions)

### display confusion matrix ###
df_cm = pd.DataFrame(
    CM,
    index = [i for i in ["Failed","Success"]],
    columns = [i for i in ["Failed","Success"]]
    )
ax = sns.heatmap(df_cm, annot=True)
ax.set(xlabel="Prediction",ylabel="Ground Truth",title="Confusion Matrix")
plt.savefig('Confusion_Matrix')
plt.show()

### print out accuracy, precision rate, and recall rate ###
print("Accuracy: ", acc)
print("Per-class precision rate: ",arrP)
print("Per-class recall rate: ",arrR)
