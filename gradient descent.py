import numpy as np
from sklearn import datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# loading datasets
dataset = datasets.load_boston()
features = dataset.data
labels = dataset.target

#Split training and testing
Nsplit = 50
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

#Define normalization function
#Define solve function
#Define error function
def preproc_params(x):
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    return mean, std

def preprocess(x,p):
    mean, std = p
    if std.any(0):
        np.put(std, np.where(std==0),1)
        x = (x-mean)/std
    else:
        x = (x-mean)/std
    return np.hstack((np.ones((np.size(x[:,0]),1)),x))

def solve(x,y):
    pinv_x = np.linalg.pinv(x)
    w = np.dot(pinv_x,y)
    return w

def mse(x,y,w):
    er = np.dot(w,x.T) - y
    err = np.inner(er,er)/np.size(y)
    return err

#normalization
params = preproc_params(X_train)
X_train = preprocess(X_train, params)
X_test = preprocess(X_test, params)

#Stochastic Gradient Descent
rate = 5e-4
epochs = 500
data_num = np.size(X_train[:,1])
feature_num = np.size(X_train[1,:])
w = np.random.uniform(-0.1,0.1,feature_num)
error = []

for i in range(epochs):
    temp = np.arange(X_train.shape[0])
    np.random.shuffle(temp)
    x = X_train[temp]
    y = y_train[temp]
    for j in range(data_num):
        diff = np.dot(w,x[j]) - y[j]
        w = w - rate*diff*x[j]
    err = mse(X_train,y_train,w)
    error.append(err)
print("the weight matrix of SGD:\n", w)
print("the final train error of SGD:\n", err)

err = mse(X_test,y_test,w)
print("the final test error of SGD\n", err)
plt.plot(range(500),error)

#Batch Gradient Descent
w = np.random.uniform(-0.1,0.1,feature_num)
error = []

for i in range(epochs):
    diff = np.dot(w,X_train.T) - y_train
    w = w - rate*np.dot(diff,X_train)
    err = mse(X_train,y_train,w)
    error.append(err)


print("the weight matrix of BGD\n", w)
print("the final train_error of BGD\n", err)
plt.plot(range(epochs),error)

err = mse(X_test,y_test,w)
print("the final test_error of BGD:\n", err)

#Closed Form Solution
w = solve(X_train, y_train)
err = mse(X_train, y_train,w)
print("the weight matrix of closed form:\n", w)

print("the train_error of closed form:\n", err)
err = mse(X_test, y_test,w)
print("the test error of cloed form\n", err)

#Testing the corectness by shuffle data selected as training or test data.
features_orig = dataset.data
labels_orig = dataset.target
Ndata = len(features_orig)

train_errs = []
test_errs = []

for k in range(100):

  # Shuffle data
  rand_perm = np.random.permutation(Ndata)
  features = [features_orig[ind] for ind in rand_perm]
  labels = [labels_orig[ind] for ind in rand_perm]

  # Train/test split
  Nsplit = 50
  X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
  X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

  # Preprocess your data - Normalization, adding a constant feature
  params = preproc_params(X_train)
  X_train = preprocess(X_train, params)
  X_test = preprocess(X_test, params)

  # Solve for optimal w
  # Use your solver function
  w = solve(X_train, y_train)

  # Collect train and test errors
  # Use your implementation of the mse function
  train_errs.append(mse(X_train, y_train, w))
  test_errs.append(mse(X_test, y_test, w))

print('Mean training error: ', np.mean(train_errs))
print('Mean test error: ', np.mean(test_errs))
