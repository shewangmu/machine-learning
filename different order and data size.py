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

def rmse(x,y,w):
    er = np.dot(w,x.T) - y
    err = np.sqrt(np.inner(er,er)/np.size(y))
    return err

#normalization
order_train = X_train
order_test = X_test

for i in range(2,5):
    order_train = np.hstack((order_train, np.power(X_train,i)))
    order_test = np.hstack((order_test, np.power(X_test,i)))

params = preproc_params(order_train)
order_train = preprocess(order_train, params)
order_test = preprocess(order_test, params)

#Computing Error of Different order
#The order range from 0 to 4, all been normolized
error_train = list()
error_test = list()
for i in range(0,5):
    data_train = order_train[:,0:13*i+1]
    data_test = order_test[:,0:13*i+1]

    w = solve(data_train,y_train)
    err = rmse(data_train,y_train,w)
    error_train.append(err)
    err = rmse(data_test,y_test,w)
    error_test.append(err)

plt.plot(range(0,5),error_train,'o')
plt.xlabel("train error")
print("train error of different order:\n", error_train)
plt.plot(range(0,5),error_test,'o')
plt.xlabel("test error")
print("test error of different order:\n", error_test)


#Different size of Training Set
#The size of data includes 20%, 40%, 60%, 80%, 100% percents of original training data
p = [0.2, 0.4, 0.6, 0.8, 1]
percent = np.dot(np.size(y_train),p)

error_train = list()
error_test = list()

for i in range(5):
    row  = int(percent[i])
    data_x = X_train[0:row,:]
    data_y = y_train[0:row]
    params = preproc_params(data_x)
    data_x = preprocess(data_x,params)
    data_test = preprocess(X_test,params)

    w = solve(data_x, data_y)
    err = rmse(data_x,data_y,w)
    error_train.append(err)
    err = rmse(data_test,y_test,w)
    error_test.append(err)

plt.plot(p, error_train, 'o')
plt.xlabel("train error with different train size")
plt.figure()
plt.plot(p, error_test, 'o')
plt.xlabel("test error with different train size")

print("train error with different train size:\n", error_train)
print("test error with different train size:\n", error_test)
