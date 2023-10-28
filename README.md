# Project_1_ML

The goal of the algorithm is to predict the likelihood of having a coronary heart disease or not. We ultimately kept the most efficient, accurate and reliable model, Regularized Logistic Regression.

FILE run.py : The main run, which produces the final predictions, using least squares and regularized logistic regression. It includes the loading, cleaning, splitting of the data, expanding of the features and prediction generation with our final model.
-> Run the code to create a prediction file.

FILE implementations.py : Mandatory functions (6 learning algorithms) with all the other required functions they need to work.

FILE functions.py : All the other functions we created to clean the data, split them, expand them, test the functions, visualize their efficiency.

FILE helpers.py : The provided functions to load the data and create a submission file.

FILE test_with_balanced_subsets.py : All the cross validation tests on each function returning their accuracy and f1-score, in order to choose the best one. Each test is first looking for the best hyperparameters in order to compare the functions based on their optimal parameters, and then returns the best parameters, the accuracy and the f1 score. Those tests are based on balanced training subsets extracted from the data.

FILE K_fold_unbalanced_data.py : k fold test of least squares followed by regularized logistic regression, using cross validation to find best hyperparameters. This test is based on unbalanced training data.

