#Test files

#In this file : Test of the efficiency of least squares followed by regularized logistic regression, using cross validation on the whole dataset without balancing the data

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

#Expansion of data
categorical_columns, numeric_columns, _, _ = separate_column_types(x_train1, max_nb=10)

x_train1=build_numerical_cos(x_train1, categorical_columns, numeric_columns)
x_test1=build_numerical_cos(x_test1, categorical_columns, numeric_columns)

#Useful functions
def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold.
    
    ------------
    Arguments:
        y: Array, shape=(N,)
        k_fold: int, K in K-fold, i.e. the fold num
        seed: int, the random seed
    ------------
    Return:
        k_indices, A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, w_initial, k_indices, k, lambda_, gamma):
    """
    Compute the loss of regularized logisitc regression for a fold corresponding to k_indices
    
    ------------
    Arguments:
        y: array, shape=(N,)
        x: array, shape=(N,D)
        k_indices: array, returned by build_k_indices()
        k: int, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_: int, hyperparameter for regularized logistic regression
        gamma: int, hyperparameter for regularized logistic regression
    -------------
    
    Returns:
        loss_tr, loss_te, train and test root mean square errors rmse = sqrt(2 mse)
    """

    # get k'th subgroup in test, others in train: 
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]

    w_final, _ = reg_logistic_regression(y_tr, x_tr, w_initial, gamma, lambda_)

    loss_te = compute_loss(y_te, x_te, w_final)
    loss_tr = compute_loss(y_tr, x_tr, w_final)

    return loss_tr, loss_te, w_final

    
def cross_validation_demo(y, x, k_fold, lambdas, gammas, w_initial):
    """
    cross validation over hyperparameters lambda and gamma.

    ------------
    Arguments:
        y: array, shape=(N,)
        x: array, shape=(N,D)
        k_fold: int, the number of folds
        lambdas: list, shape = (p, ) where p is the number of values of lambda to test
        gammas: list, shape = (q, ) where q is the number of values of gamma to test
        w_initial: array, shape (,D), initial weights for regularized logistic regression
    ------------
    
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """

    seed = 12
    k_fold = k_fold
    lambdas = lambdas
    gammas = gammas
    weights = []
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    losses_te = np.zeros((len(lambdas),len(gammas)))
    losses_tr = np.zeros((len(lambdas),len(gammas)))
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation over lambdas and gammas:
    for i, lambda_ in enumerate (lambdas):
        for j, gamma in enumerate (gammas): 
            rmse_tr_lambdai_gammaj=[]
            rmse_te_lambdai_gammaj=[]
            w = w_initial
            for k in range (k_fold):
                rmse_tr_lambdai_gammaj.append(cross_validation(y, x, w, k_indices, k, lambda_, gamma)[0])
                rmse_te_lambdai_gammaj.append(cross_validation(y, x, w, k_indices, k, lambda_, gamma)[1])
                weights.append(cross_validation(y, x, w, k_indices, k, lambda_, gamma)[2])
                
            losses_tr[i][j] = np.mean(rmse_tr_lambdai_gammaj)
            losses_te[i][j] = np.mean(rmse_te_lambdai_gammaj)
 

    best_index = np.unravel_index(np.argmin(losses_te), losses_te.shape)
    row_index, col_index = best_index
    best_weight = weights[row_index][col_index]
    print('Best lambda : ', lambdas[row_index])
    print('Best gamma : ', gammas[col_index])
    print('Smallest loss : ', losses_te[best_index])

    return lambdas[row_index], gammas[col_index], best_weight
    
    
#Initial w using least squares
w_initial,_ = least_squares(y_train1, x_train1)

#cross validation
lambda_, gamma, w = cross_validation_demo(y_train1, x_train1, 3, np.linspace(0,0.3,10), np.linspace(0,0.3,10), w_initial)

#Test of efficiency
nb_steps = 150

w = w_initial
for i in range(nb_steps):
    w, loss= reg_logistic_regression(y_train1, x_train1, w, gamma, lambda_)

w_final = w

test(w_final, x_train1, y_train1)



