# machine-learning-project-boston-housing-dataset
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from IPython.display import HTML

#Load the dataset
boston=load_boston()

#Description of the dataset
print(boston.DESCR)
#Put the data into pandas DataFrames
features = pd.DataFrame(boston.data,columns=boston.feature_names)
features

target = pd.DataFrame(boston.target,columns=['target'])
target

max(target['target'])

min(target['target'])
#Concatenate features and target into a single DataFrame
#axis = 1 makes it concatenate column wise
df=pd.concat([features,target],axis=1)
df

#summary of dataset
#Use round(decimals=2) to set the precision to 2 decimal places
df.describe().round(decimals = 2)

#Calculate correlation between every column on the data
corr = df.corr('pearson')

#Take absolute values of correlations
corrs = [abs(corr[attr]['target']) for attr in list(features)]

#Make a list of pairs [(corr,feature)]
l = list(zip(corrs,list(features)))

#Sort the list of pairs in reverse/descending order,
#with the correlation value as the key for sorting
l.sort(key = lambda x : x[0], reverse=True)

#"Unzip" pairs to two lists
#Zip(*l) - takes a list that looks like [[a,b,c], [d,e,f], [g,h,i]]
#and return [[a,d,g], [b,e,h], [c,f,i]]
corrs, labels = list(zip((*l)))

#Plot correlations with respect to the target variable as a bar gragh
index = np.arange(len(labels))
plt.figure(figsize=(15,5))
plt.bar(index, corrs, width=0.5)
plt.xlabel('Attributes')
plt.ylabel('Correlation with the target variable')
plt.xticks(index, labels)
plt.show()

X=df['LSTAT'].values
Y=df['target'].values

x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X.reshape(-1, 1))
X = X[:, -1]
y_scaler = MinMaxScaler()
Y = y_scaler.fit_transform(Y.reshape(-1, 1))
Y = Y[:, -1]


#0.2 indicates 20% of the data is randomly sampled as testing data
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)

def error(m, x, c, t):
    N = x.size
    e = sum(((m * x + c) - t) ** 2)
    return e * 1/(2 * N)

def update(m, x, c, t, learning_rate):
    grad_m = sum(2* ((m * x + c) - t) * x)
    grad_c = sum(2* ((m * x + c) - t))
    m = m - grad_m * learning_rate
    c = c - grad_c * learning_rate
    return m, c
    def gradient_descent(init_m, init_c, x, t, learning_rate, iterations, error_threshold):
    m = init_m
    c = init_c
    error_values = list()
    mc_values = list()
    for i in range(iterations):
        e = error(m, x, c, t)
        if e < error_threshold:
            print('Error less than the thresold. Stopping gradient descent')
            break
        error_values.append(e)
        m, c = update(m, x, c, t, learning_rate)
        mc_values.append((m,c))
    return m, c, error_values, mc_values
    
    %%time
init_m = 0.9
init_c = 0
learning_rate = 0.001
iterations = 250
error_threshold = 0.001


m, c, error_values, mc_values = gradient_descent(init_m, init_c, xtrain, ytrain, learning_rate, iterations, error_threshold)

print(m ,c)
plt.scatter(xtrain, ytrain, color='b')
plt.plot(xtrain, (m * xtrain + c) , color='r')

plt.plot(np.arange(len(error_values)), error_values)
plt.ylabel('Error')
plt.xlabel('Iterations')

# Calculate the predictions on the test set as a vectorized operation
predicted = (m * xtest) + c

# Compute MSE for the predicted values on the testing set
mean_squared_error(ytest, predicted)

# Put xtest, ytest and predicted values into a single DataFrame so that we
# can see the predicted values alongside the testing set
p = pd.DataFrame(list(zip(xtest, ytest, predicted)), columns=['x', 'target_y', 'predicted_y'])


plt.scatter(xtest, ytest, color='b')
plt.plot(xtest, predicted, color='r')
# Reshape to change the shape that is required by the scaler
predicted = predicted.reshape(-1, 1)
xtest = xtest.reshape(-1, 1)
ytest = ytest.reshape(-1, 1)

xtest_scaled = x_scaler.inverse_transform(xtest)
ytest_scaled = y_scaler.inverse_transform(ytest)
predicted_scaled = y_scaler.inverse_transform(predicted)

# This is to remove the extra dimension
xtest_scaled = xtest_scaled[:, -1]
ytest_scaled = ytest_scaled[:, -1]
predicted_scaled = predicted_scaled[:, -1]

p = pd.DataFrame(list(zip(xtest_scaled, ytest_scaled, predicted_scaled)), columns=['x', 'target_y', 'predicted_y'])
p = p.round(decimals = 2)
p.head()
