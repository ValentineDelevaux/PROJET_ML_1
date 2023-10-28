# Project_1_ML

FILE run.py : The main run, which produces the final predictions, using least squares and regularized logistic regression. Tt includes the loading, cleaning, splitting of the data, expanding of the features and prediction generation with our final model.
-> Run the code to create a prediction file.

FILE implementations.py : Mandatory functions (6 learning algorithms) with all the other required functions they need to work.

FILE functions.py : All the other functions we created to clean the data, split them ,expand them, test the functions, visualize their efficiency.

FILE helpers.py : The given functions to load the data and create a submission file.

FILE test_with_balanced_subsets.py : All the tests on each function returning their accuracy and f1-score, in order to choose the best one. Each test is first looking for the best hyperparameters in order to compare the functions based on their optimal parameters, and then returns the best parameters, the accuracy and the f1 score. Those tests are based on balanced subsets extracted from the data.

FILE test_K_fold_unbalanced_dataset.py : Test of least squares followed by regularized logistic regression, using cross validation to find best hyperparameters. This test is based on unbalanced data.

