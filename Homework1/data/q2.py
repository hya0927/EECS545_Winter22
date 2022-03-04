import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math


X = np.load('q2x.npy')
y = np.load('q2y.npy')
# print(X)
# print(y)
X = np.reshape(X, (100,1))
y = np.reshape(y, (100,1))

plt.figure()
plt.scatter(X,y)

def generate_fai(X, degree):
    X_b = np.ones(len(X))
    X_b = np.reshape(X_b, (len(X),1))
    for i in range(1,degree+1):
        X_b = np.c_[X_b, np.power(X, i)]
    return X_b

def generate_W(X, y, degree):
    X_b = np.ones(len(X))
    X_b = np.reshape(X_b, (len(X),1))
    for i in range(1,degree+1):
        X_b = np.c_[X_b, np.power(X, i)]
    W_1 = np.linalg.inv(np.dot(X_b.T, X_b))
    W_2 = np.dot(X_b.T, y)
    W = np.dot(W_1, W_2)
    return W

def generate_weight_R(X, evaluating_point, width):
    M = len(X)
    R = np.zeros((M,M))
    X_i = X[evaluating_point]*np.ones(M)
    X_i = np.reshape(X_i,(M,1))
    A = -np.square(X - X_i)
    B = 2*np.square(width)
    C = np.exp(A/B)
    for i in range(M):
        R[i][i] = C[i]
    #R = math.exp(-np.square(X_expand - X_fai)/(2*np.square(width)))
    return R

X_fai_closedform= generate_fai(X, 1)
W_closedform = generate_W(X, y, 1)

width = 0.8
degree = 1
# evaluating_point = 50
X_fai = generate_fai(X,degree)
prediction = np.zeros(len(X))
for i in range(len(X)):
    evaluating_point = i
    R = generate_weight_R(X, evaluating_point, width)
    W_1 = np.dot(X_fai.T, R)
    W_2 = np.linalg.inv(np.dot(W_1, X_fai))
    W_3 = np.dot(X_fai.T, R)
    W_4 = np.dot(W_3, y)
    W_weight = np.dot(W_2, W_4)
    prediction[i] = np.dot(X_fai_closedform[evaluating_point,:], W_weight)

y_plot = np.dot(X_fai_closedform, W_closedform)

plt.figure()
plt.scatter(X, y, label='Train')
plt.plot(X, y_plot, 'r-', label='unweighted linear regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('2_d1.png')
print("W_closedform0: {:0.3f},\nW_closedform1:{:0.3f}".format(W_closedform[0][0], W_closedform[1][0]))

def generate_weight_R_refine(X, evaluating_point, width):
    M = len(X)
    R = np.zeros((M,M))
    X_i = X[evaluating_point]*np.ones(M)
    X_i = np.reshape(X_i,(M,1))
    print(X_i.shape)
    A = -np.square(X - X_i)
    B = 2*np.square(width)
    C = np.exp(A/B)
    for i in range(M):
        R[i][i] = C[i]
    #R = math.exp(-np.square(X_expand - X_fai)/(2*np.square(width)))
    return R

width = 0.8
degree = 1
#evaluating_point = 50
X_fai = generate_fai(X,degree)
prediction = np.zeros(len(X))
for i in range(len(X)):
    evaluating_point = i
    R = generate_weight_R(X, evaluating_point, width)
    W_1 = np.dot(X_fai.T, R)
    W_2 = np.linalg.inv(np.dot(W_1, X_fai))
    W_3 = np.dot(X_fai.T, R)
    W_4 = np.dot(W_3, y)
    W_weight = np.dot(W_2, W_4)
    prediction[i] = np.dot(X_fai_closedform[evaluating_point,:], W_weight)

plt.figure()
plt.scatter(X, y, label='Train')
plt.plot(X, y_plot, 'r-', label='unweighted linear regression')
plt.scatter(X, prediction, label='width=0.8')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('2_d2.png')

width = 0.1
degree = 1
#evaluating_point = 50
X_fai = generate_fai(X,degree)
prediction_01 = np.zeros(len(X))
for i in range(len(X)):
    evaluating_point = i
    R = generate_weight_R(X, evaluating_point, width)
    W_1 = np.dot(X_fai.T, R)
    W_2 = np.linalg.inv(np.dot(W_1, X_fai))
    W_3 = np.dot(X_fai.T, R)
    W_4 = np.dot(W_3, y)
    W_weight = np.dot(W_2, W_4)
    prediction_01[i] = np.dot(X_fai_closedform[evaluating_point,:], W_weight)

width = 0.3
degree = 1
#evaluating_point = 50
X_fai = generate_fai(X,degree)
prediction_03 = np.zeros(len(X))
for i in range(len(X)):
    evaluating_point = i
    R = generate_weight_R(X, evaluating_point, width)
    W_1 = np.dot(X_fai.T, R)
    W_2 = np.linalg.inv(np.dot(W_1, X_fai))
    W_3 = np.dot(X_fai.T, R)
    W_4 = np.dot(W_3, y)
    W_weight = np.dot(W_2, W_4)
    prediction_03[i] = np.dot(X_fai_closedform[evaluating_point,:], W_weight)

width = 2
degree = 1
#evaluating_point = 50
X_fai = generate_fai(X,degree)
prediction_2 = np.zeros(len(X))
for i in range(len(X)):
    evaluating_point = i
    R = generate_weight_R(X, evaluating_point, width)
    W_1 = np.dot(X_fai.T, R)
    W_2 = np.linalg.inv(np.dot(W_1, X_fai))
    W_3 = np.dot(X_fai.T, R)
    W_4 = np.dot(W_3, y)
    W_weight = np.dot(W_2, W_4)
    prediction_2[i] = np.dot(X_fai_closedform[evaluating_point,:], W_weight)

width = 10
degree = 1
#evaluating_point = 50
X_fai = generate_fai(X,degree)
prediction_10 = np.zeros(len(X))
for i in range(len(X)):
    evaluating_point = i
    R = generate_weight_R(X, evaluating_point, width)
    W_1 = np.dot(X_fai.T, R)
    W_2 = np.linalg.inv(np.dot(W_1, X_fai))
    W_3 = np.dot(X_fai.T, R)
    W_4 = np.dot(W_3, y)
    W_weight = np.dot(W_2, W_4)
    prediction_10[i] = np.dot(X_fai_closedform[evaluating_point,:], W_weight)

plt.figure()
fig, ax = plt.subplots(figsize=(16, 10))
plt.subplot(2, 2, 1)
plt.scatter(X, y, label='Train')
plt.plot(X, y_plot, 'r-', label='unweighted linear regression')
plt.scatter(X, prediction_01, label='width=0.1')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.subplot(2, 2, 2)
plt.scatter(X, y, label='Train')
plt.plot(X, y_plot, 'r-', label='unweighted linear regression')
plt.scatter(X, prediction_03, label='width=0.3')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.subplot(2, 2, 3)
plt.scatter(X, y, label='Train')
plt.plot(X, y_plot, 'r-', label='unweighted linear regression')
plt.scatter(X, prediction_2, label='width=2')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.subplot(2, 2, 4)
plt.scatter(X, y, label='Train')
plt.plot(X, y_plot, 'r-', label='unweighted linear regression')
plt.scatter(X, prediction_10, label='width=10')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('2_d3.png', dpi=600)