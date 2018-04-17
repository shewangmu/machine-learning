import numpy as np
import matplotlib.pyplot as plt

def data_generator(size,noise_scale=0.05):
    xs = np.random.uniform(low=0,high=3,size=size)

    # for function y = 0.5x - 0.3 + sin(x) + epsilon, where epsilon is a gaussian noise with std dev= 0.05
    ys = xs * 0.5 - 0.3 + np.sin(3*xs) + np.random.normal(loc=0,scale=noise_scale,size=size)
    return xs, ys

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

def solve_0(x,y):
    pinv_x = np.linalg.pinv(x)
    w = np.dot(pinv_x,y)
    return w

def solve_1(x,y,r):
    m = np.linalg.pinv(np.dot(xtr.T,np.sqrt(r)))
    n = np.dot(np.sqrt(r),y_train)
    w = np.dot(m.T,n)
    return w

def generate_r(xn,x,tao):
    r = []
    for i in range(np.size(xn[:,1])):
        norm = np.inner(xn[i]-x, xn[i]-x)
        rn = np.exp(-norm/(2*tao**2))
        r.append(rn)
    r = np.diag(r)
    return r

def mse(x,y,w):
    er = np.dot(x,w) - y
    err = np.inner(er.T,er.T)/np.size(y)
    return err

def main():
    noise_scales = [0.05,0.2]
    noise_scale = noise_scales[0]
    X_train, y_train = data_generator((100,1),noise_scale=noise_scale)
    X_test, y_test = data_generator((30,1),noise_scale=noise_scale)
    sigma_paras = [0.1,0.2,0.4,0.8,1.6]

    #preprocess
    p = preproc_params(X_train)
    xtr = preprocess(X_train,p)
    xte = preprocess(X_test,p)

    #linear regression
    w = solve_0(xtr,y_train)
    err = rmse(xte,y_test,w)
    print("the test rmse is:", float(err))
    y_count = np.dot(xte,w)
    plt.plot(X_test,y_count,'o',label="predicted")
    plt.plot(X_test,y_test,'o',label="true")
    plt.legend()
    plt.title("predicted labels")

    #parameter = 0.2
    y = []
    err = 0
    for i in range(np.size(xte[:,1])):
        r = generate_r(xtr,xte[i,:],0.2)
        w = solve_1(xtr,y_train,r)
        y_count = float(np.dot(xte[i,:],w))
        y.append(y_count)
        err += (y_count-y_test[i])**2
    print("test error for parameter = 0.2:",np.sqrt(err)/30)
    plt.plot(X_test,y,'o',"predicted")
    plt.plot(X_test,y_test,'o',"true")
    plt.legend()
    plt.title("predicted label for 0.2")

    #parameter = 2
    y = []
    err = 0
    for i in range(np.size(xte[:,1])):
        r = generate_r(xtr,xte[i,:],2)
        w = solve_1(xtr,y_train,r)
        y_count = np.dot(xte[i,:],w)
        y.append(y_count)
        err += (y_count-y_test[i])**2
    print("test errors for parameter = 2:",np.sqrt(err)/30)
    plt.plot(X_test,y,'o',label='predicted')
    plt.plot(X_test,y_test,'o',label="true")
    plt.legend()
    plt.title("predicted label for 2")

main()
