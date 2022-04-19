# -*- coding: utf-8 -*-
"""*Ensemble Classifiers*

*André Paulo Ferreira Machado*

This work consists of carrying out an experimental comparison between a pre-defined set of learning techniques
for automatic classification, based on the idea of combined classifiers, applied to some
classification problems. The chosen techniques are: Bagging, AdaBoost, RandomForest and HeterogeneousPooling.

Databases used: digits, wine e breast cancer.
"""

"""The results of each classifier are presented in a table containing the average of the accuracies
obtained in each fold of the external cycle, the standard deviation and the confidence interval at 95% of significance
of the results, and also through the boxplot of the results of each classifier in each fold.

The data used in the training set in each test run are standardized (normalized or z-score).
The standardization values ​​obtained from the training data are used to standardize the data of the respective test set.
The experimental procedure of training, validation and testing is carried out through 3 rounds of nested cycles
validation and testing cycle, with the internal validation cycle containing 4 folds and the external testing cycle containing 10 folds.
The grid search of the internal loop considers the hyperparameter values ​​defined for each learning technique.
"""

from sklearn import datasets

import inquirer


"""# Evatuate Classifiers"""
import evaluateClassifiers as evaluate


"""# DataBase Selection"""
questions = [
  inquirer.List('base',
                message="Select one DataBase",
                choices=['Digits', 'Wine', 'Breast Cancer'],
            ),
]
answers = inquirer.prompt(questions)

def functionDigits():
    print("DataBase Digits selected")
    dataBase = datasets.load_digits()
    return dataBase

def functionWine():
    print("DataBase Wine selected")
    dataBase = datasets.load_wine()
    return dataBase

def functionBreast():
    print("DataBase Breast Cancer selected")
    dataBase = datasets.load_breast_cancer()
    return dataBase

def default():
    print("Select one DataBase")

if __name__ == "__main__":
    switch = {
        "Digits": functionDigits,
        "Wine": functionWine,
        "Breast Cancer": functionBreast
    }

    case = switch.get(answers["base"], default)
    dataBase = case()
    evaluate.evaluateClassifiers(dataBase)
    
