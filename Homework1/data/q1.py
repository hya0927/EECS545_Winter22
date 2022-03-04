import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math


# cost function
def cal_cost(W, X, y):
    m = len(y)
    predictions = np.dot(X, W)
    cost = (1 / (m)) * np.sum(np.square(predictions - y))
    return cost


# load training set
X = np.load('q1xTrain.npy')
y = np.load('q1yTrain.npy')
# print(X)
# print(y)
X = np.reshape(X, (20, 1))
y = np.reshape(y, (20, 1))
# print(X.shape)
# print(y.shape)

# show scatter of training set
plt.figure()
plt.scatter(X, y)


# batch gradient descent
def batch_gradient_descent(X, y, W, learning_rate=1e-3, iterations=100):
    m = len(y)
    y = np.reshape(y, (20, 1))
    cost_history = np.zeros(iterations)
    W_history = np.zeros((iterations, 2))
    for i in range(iterations):
        prediction = np.dot(X, W)
        W = W - learning_rate * np.dot(X.T, prediction - y)
        W_history[i, 0] = W[0][0]
        W_history[i, 1] = W[1][0]
        cost_history[i] = cal_cost(W, X, y)
    # return w, some_additional_data
    return W, cost_history, W_history


# initial value of coefficient

W_random = np.random.randn(2, 1)

# hyperparameter of batch gradient descent
lr = 0.05
n_iter = 200
# print(W_random)
# W = np.zeros(2)
# W = np.reshape(W, (2,1))
X_b = np.c_[np.ones((len(X), 1)), X]
W_BGD, cost_history, W_history = batch_gradient_descent(X_b, y, W_random, lr, n_iter)
print("W_BGD0: {:0.3f},\nW_BGD1:{:0.3f}".format(W_BGD[0][0], W_BGD[1][0]))
print("Final cost/E_MS:  {:0.3f}".format(cost_history[-1]))

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.set_ylabel('Cost')
# ax.set_xlabel('Iterations')
# _ = ax.plot(range(n_iter), cost_history, 'b.')

y_plot = np.dot(X_b, W_BGD)
# plt.scatter(X,y, label='Train')
# plt.scatter(X, y_plot, label='BGD')
# plt.legend()
# print("W_BGD0: {:0.3f},\nW_BGD1:{:0.3f}".format(W_BGD[0][0], W_BGD[1][0]))
# print("Final cost/E_MS:  {:0.3f}".format(cost_history[-1]))

# stocashtic gradient descent
def stocashtic_gradient_descent_2(X, y, W, learning_rate=0.01, iterations=10):
    m = len(y)
    cost_history = np.zeros(iterations)
    error_history = np.zeros(iterations)
    for itera in range(iterations):
        cost = 0.0
        for i in range(m):
            X_i = X[i, :].reshape(1, X.shape[1])
            # print(X_i)
            y_i = y[i].reshape(1, 1)
            # print(y_i)
            prediction = np.dot(X_i, W)
            # print(prediction)
            W = W - learning_rate * np.dot(X_i.T, prediction - y_i)
            cost += cal_cost(W, X_i, y_i)
            # print(cost)
        cost_history[itera] = cost/m
        error_history[itera] = cal_cost(W, X, y)
    return W, cost_history, error_history

lr = 0.05
n_iter = 200
# W_SGD_2 = np.random.randn(2, 1)
X_b = np.c_[np.ones((len(X),1)),X]
W_SGD_2, cost_history_SGD_2, error_history_SGD_2 = stocashtic_gradient_descent_2(X_b,y,W_random,lr,n_iter)
print("W_SGD_2 0: {:0.3f},\nW_SGD_2 1:{:0.3f}".format(W_SGD_2[0][0], W_SGD_2[1][0]))
print("Final cost/MSE:  {:0.3f}".format(cost_history_SGD_2[-1]))
print("Final E_MS:  {:0.3f}".format(error_history_SGD_2[-1]))

plt.figure()
fig, ax = plt.subplots(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(range(n_iter), cost_history, 'b.', label='BGD')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(n_iter), error_history_SGD_2, 'y.', label='SGD')
plt.legend()
plt.savefig('Cost_BGD_SGD.png')


plt.figure()
y_plot_SGD = np.dot(X_b, W_SGD_2)
y_plot_BGD = np.dot(X_b, W_BGD)

plt.scatter(X, y, label='Train')
plt.scatter(X, y_plot_SGD, label='SGD')
plt.scatter(X, y_plot_BGD, label='BGD')
plt.legend()
# print("W_SGD_2 0: {:0.3f},\nW_SGD_2 1:{:0.3f}".format(W_SGD_2[0][0], W_SGD_2[1][0]))
# print("Final E_MS:  {:0.3f}".format(error_history_SGD_2[-1]))
plt.savefig('fitting_BGD_SGD.png')

# closedform
def closed_form_solution(X, y, degree):
    X_b = np.ones(len(X))
    X_b = np.reshape(X_b, (len(X),1))
    for i in range(1,degree+1):
        X_b = np.c_[X_b, np.power(X, i)]
    W_1 = np.linalg.inv(np.dot(X_b.T, X_b))
    W_2 = np.dot(X_b.T, y)
    W = np.dot(W_1, W_2)
    return W, X_b

M = len(X)
degree = 9
E_RMS_train = np.zeros(degree+1)
y_mean = np.sum(y)/M
for i in range(degree+1):
    W_closedform, X_fai_closedform = closed_form_solution(X,y,i)
    predictions = np.dot(X_fai_closedform, W_closedform)
    cost = (1/2) * np.sum(np.square(predictions-y))
    E_RMS = math.sqrt(2*cost/len(X))
    E_RMS = E_RMS
    E_RMS_train[i] = E_RMS
    # print(E_RMS)

# load test set
X_test = np.load('q1xTest.npy')
y_test = np.load('q1yTest.npy')
X_test = np.reshape(X_test, (20,1))
y_test = np.reshape(y_test, (20,1))

M = len(X)
degree = 9
E_RMS_test = np.zeros(degree+1)
y_mean = np.sum(y)/M
for i in range(degree+1):
    W_closedform, X_fai_closedform = closed_form_solution(X,y,i)
    W_test, X_b1 = closed_form_solution(X_test,y_test,i)
    predictions = np.dot(X_b1, W_closedform)
    cost = (1/2) * np.sum(np.square(predictions-y_test))
    # print(cost*2/len(X_test))
    E_RMS = math.sqrt(2*cost/len(X_test))
    E_RMS_test[i] = E_RMS


x = np.zeros(degree+1)
for i in range(degree+1):
    x[i] = i
#print(x)
plt.figure()
plt.plot(x,E_RMS_train, 'bo-', label='Training')
plt.plot(x,E_RMS_test, 'ro-', label='Test')
plt.xlabel('M')
plt.ylabel('E_{RMS}')
plt.legend()
plt.savefig('overfitting.png')


def closed_form_solution_regu(X, y, degree, lamda):
    X_b = np.ones(len(X))
    X_b = np.reshape(X_b, (len(X),1))
    for i in range(1,degree+1):
        X_b = np.c_[X_b, np.power(X, i)]
    W_1 = np.linalg.inv(np.dot(X_b.T, X_b) + lamda*np.eye(degree+1))
    W_2 = np.dot(X_b.T, y)
    W = np.dot(W_1, W_2)
    return W, X_b

M_lamda = 10
M = len(X)
E_RMS_train_regu = np.zeros(M_lamda)
for i in range(M_lamda):
    if(i == 0):
        lamda = 0
    else:
        lamda = 1/np.power(10,M_lamda-1-i)
    # print(lamda)
    W_train_regu, X_b_train_regu = closed_form_solution_regu(X,y,9,lamda)
    predictions = np.dot(X_b_train_regu, W_train_regu)
    cost_train_regu = (1/2) * np.sum(np.square(predictions-y))
    E_RMS = math.sqrt(2*cost_train_regu/len(X))
    E_RMS_train_regu[i] = E_RMS

M = len(X)
E_RMS_test_regu = np.zeros(M_lamda)
for i in range(M_lamda):
    if(i == 0):
        lamda = 0
    else:
        lamda = 1/np.power(10,M_lamda-1-i)
    # print(lamda)
    W_train_regu, X_b_train_regu = closed_form_solution_regu(X,y,M_lamda-1,lamda)
    W_test, X_b_test_regu = closed_form_solution_regu(X_test,y_test,M_lamda-1,lamda)
    predictions = np.dot(X_b_test_regu, W_train_regu)
    cost_regu_test = (1/2) * np.sum(np.square(predictions-y_test))
    E_RMS = math.sqrt(2*cost_regu_test/len(X))
    E_RMS_test_regu[i] = E_RMS

x = np.zeros(M_lamda)
for i in range(M_lamda):
    if(i == 0):
        x[i] = -M_lamda
    else:
        x[i] = -(M_lamda-1)+i
#print(x)
plt.figure()
plt.plot(x,E_RMS_train_regu, 'bo-', label='Training')
plt.plot(x,E_RMS_test_regu, 'ro-', label='Test')
plt.xlabel('ln_{lamda}')
plt.ylabel('E_{RMS}')
plt.legend()
plt.savefig('regu.png')

