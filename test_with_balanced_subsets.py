#Test files

#In this file : All the tests on every functions we have in the file implementations.py. The goal was to test all functions and see which one is the more efficient for our dataset. To do that, we computed the optimized hyperparameters for each, and found the accuracy and the f1 score. This allows us to choose the best option for our final algorithm.

#Importation of other files
from implementations import *
from helpers import *
import matplotlib.pyplot as plt
import numpy as np
from functions import *

#Loading of data
x_train, x_test, y_train, train_ids, test_ids = load_csv_data('dataset_to_release/', sub_sample=False)

#Cleaning of data
x_train1, y_train1, x_test1 = clean_data(x_train, y_train, x_test)

#Extansion of data
categorical_columns, numeric_columns, _, _ = separate_column_types(x_train1, max_nb=10)
x_train1=build_numerical_cos(x_train1, categorical_columns, numeric_columns)
x_test1=build_numerical_cos(x_test1, categorical_columns, numeric_columns)

#Spliting of data
x_tr, y_tr, x_te, y_te = split_data(x_train1, y_train1)

#Creation of balanced subsets
sub_y_trains, sub_x_trains = creation_subsets(y_tr, x_tr)

#-------------------------------------------------------------
#Test of Least Squares
weights_LS = []

for i in range(len(sub_x_trains)):
    w,_ = least_squares(sub_y_trains[i], sub_x_trains[i])
    weights_LS.append(w)
    
w_LS = np.mean(weights_LS, axis = 0)

print('Test of Least Squares : ')
test(w_LS, x_te, y_te)

#-------------------------------------------------------------
#Test of Ridge Regression
lambdas = np.linspace(0,0.001,20)  
weights_RIDGE = []
losses_te = []
acc = []
f = []

for lambda_ in lambdas:
    weights_lambda = []
    for i in range(len(sub_x_trains)):
        w,_ = ridge_regression(sub_y_trains[i], sub_x_trains[i], lambda_)
        weights_lambda.append(w)
        
    weights_RIDGE.append(np.mean(weights_lambda, axis = 0))
    loss = compute_loss(y_te, x_te, np.mean(weights_lambda, axis = 0))
    losses_te.append(loss)
    f.append(compute_f1(y_te, sigmoid_prediction_ (x_te, np.mean(weights_lambda, axis = 0), threshold=0.59)))
    acc.append(accuracy(y_te, sigmoid_prediction_ (x_te, np.mean(weights_lambda, axis = 0), threshold=0.59)))

print('Test of Ridge Regression: ')
    
best_index = f.index(max(f))
print('Smallest loss : ',losses_te[best_index])
print('Best lambda : ',lambdas[best_index])
w_RIDGE = weights_RIDGE[best_index]

test(w_RIDGE, x_te, y_te)


#-------------------------------------------------------------
#We use the w found with least squares to begin the gradient descents (ridge regression is not better if the best lambda = 0)
w_initial = w_LS

#-------------------------------------------------------------
#Test of Mean Squared Error Gradient Descent
losses_te = []
weights_MS = []
f = []
acc = []
max_iters = 10
gammas = np.linspace(0,1,50)

for gamma in gammas :
    weights_sub = []
    for i in range(len(sub_x_trains)):
        w = w_initial
        w, loss = mean_squared_error_gd(sub_y_trains[i], sub_x_trains[i], w, max_iters, gamma)
        weights_sub.append(w)
    weights_MS.append(np.mean(weights_sub, axis = 0))
    losses_te.append(compute_loss(y_te, x_te, np.mean(weights_sub, axis = 0)))
    f.append(compute_f1(y_te, sigmoid_prediction_ (x_te, np.mean(weights_sub, axis = 0), threshold=0.6)))
    acc.append(accuracy(y_te, sigmoid_prediction_ (x_te, np.mean(weights_sub, axis = 0), threshold=0.6)))

print('Test of Mean squared error gradient descent: ')    
    
best_index = f.index(max(f))
print('Best gamma : ', gammas[best_index])
print('Smallest loss : ', losses_te[best_index])
w_final = weights_MS[best_index]   

test(w_final, x_te, y_te)


#-------------------------------------------------------------
#Test of Mean Squared Error Stochastic Gradient Descent
losses_te = []
weights_SMS = []
f = []
acc = []
max_iters = 10
batch_size = 20000
gammas = np.linspace(0,1,50)

for gamma in gammas :
    weights_sub = []
    for i in range(len(sub_x_trains)):
        w = w_initial
        w,_ = mean_squared_error_sgd(sub_y_trains[i], sub_x_trains[i], w, batch_size, max_iters, gamma)
        weights_sub.append(w)
    weights_SMS.append(np.mean(weights_sub, axis = 0))    
    losses_te.append(compute_loss(y_te, x_te, np.mean(weights_sub, axis = 0)))
    f.append(compute_f1(y_te, sigmoid_prediction_(x_te, np.mean(weights_sub, axis = 0), threshold=0.6)))
    acc.append(accuracy(y_te, sigmoid_prediction_ (x_te, np.mean(weights_sub, axis = 0), threshold=0.6)))

print('Test of Mean squared error stochastic gradient : ')    
    
best_index = losses_te.index(min(losses_te))
print('Best gamma : ', gammas[best_index])
print('Smallest loss : ', losses_te[best_index])
w_SMS = weights_SMS[best_index]   

test(w_SMS, x_te, y_te)


#-------------------------------------------------------------
#Test of Logistic Regression
losses_te = []
weights_LOG = []
f = []
acc = []

nb_steps = 15
gammas = np.linspace(0,1,50)

for gamma in gammas :
    weights_gamma = []
    for i in range(len(sub_x_trains)):
        w = w_initial
        
        for j in range(nb_steps):
            w,_ = logistic_regression(sub_y_trains[i], sub_x_trains[i], w, gamma)
            
        weights_gamma.append(w)
        
    weights_LOG.append(np.mean(weights_gamma, axis = 0))
    losses_te.append(compute_loss(y_te, x_te, np.mean(weights_gamma, axis = 0)))
    f.append(compute_f1(y_te, sigmoid_prediction_(x_te, np.mean(weights_gamma, axis = 0), threshold=0.6)))
    acc.append(accuracy(y_te, sigmoid_prediction_ (x_te, np.mean(weights_gamma, axis = 0), threshold=0.6)))
 

print('Test of Logistic Regression : ')
    
best_index = f.index(max(f))
print('Best gamma : ', gammas[best_index])
print('Smallest loss : ', losses_te[best_index])
w_LOG = weights_LOG[best_index]  

test(w_LOG, x_te, y_te) 

#-------------------------------------------------------------
#Test of Regularized Logistic Regression
nb_steps = 15
lambdas = np.linspace(0,0.1,10)
gammas = np.linspace(0,0.2,10)
weights_RLOG = np.zeros((len(lambdas),len(gammas), x_tr.shape[1]))
losses_te = np.zeros((len(lambdas),len(gammas)))
losses_tr = np.zeros((len(lambdas),len(gammas)))

for i, lambda_ in enumerate(lambdas):
    for j, gamma in enumerate(gammas):
        weights_sub = []
        for a in range(len(sub_x_trains)):
            w = w_initial
            for step in range(nb_steps):

                w, _ = reg_logistic_regression(sub_y_trains[a], sub_x_trains[a], w, gamma, lambda_)
            weights_sub.append(w)

        weights_RLOG[i,j] = np.mean(weights_sub, axis = 0)
        loss_te = compute_loss(y_te, x_te, np.mean(weights_sub, axis = 0))
        loss_tr = compute_loss(y_tr, x_tr, np.mean(weights_sub, axis = 0))
        losses_te[i][j] = loss_te 
        losses_tr[i][j] = loss_tr 

print('Test of regularized logistic regression :')        
best_index = np.unravel_index(np.argmin(losses_te), losses_te.shape)
row_index, col_index = best_index
best_weight = weights_RLOG[row_index][col_index]
print('Best lambda : ', lambdas[row_index])
print('Best gamma : ', gammas[col_index])
print('Smallest loss : ', losses_te[best_index])
w_RLOG = best_weight

test(w_RLOG, x_te, y_te)

