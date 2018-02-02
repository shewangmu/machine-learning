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

Ns = round(np.size(X_train[:,0])*0.1)
new_x, new_y = X_train[:-Ns], y_train[:-Ns]
val_x, val_y = X_train[-Ns:], y_train[-Ns:]


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

#Normalization
lam = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
params = preproc_params(new_x)
new_x = preprocess(new_x, params)
X_test = preprocess(X_test, params)
val_x = preprocess(val_x, params)

#Error of Different Lambda
error_train = list()
error_val = list()
m = np.size(new_y)
n = np.size(new_x[0,:])
w0 = []

for i in range(6):
#Create the new data
    data_x = new_x/m**0.5
    data_x = np.append(data_x, np.identity(n)*lam[i]**0.5)
    data_x = data_x.reshape(m+n,n)
    data_y = new_y/m**0.5
    data_y = np.append(data_y, np.zeros(n))

    w = solve(data_x, data_y)
    err = rmse(new_x, new_y, w)
    error_train.append(err)
    err = rmse(val_x, val_y, w)
    error_val.append(err)
    w0.append(w)



w0 = np.array(w0)
w0 = w0.reshape(6,np.size(w))

plt.plot(lam, error_train)
plt.figure()
plt.plot(lam, error_val)

print("the validation error with different lambda:\n", error_val)

w = w0[3,:]
err = rmse(X_test,y_test,w)
print("the test error when lambda=0.3 is:\n", err)
