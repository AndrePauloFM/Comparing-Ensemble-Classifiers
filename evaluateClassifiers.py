from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

"""# Class - Heterogeneous Polling"""
import heterogeneousClassifier as HP

import gridTest
import getResults


dfResultClassifier = pd.DataFrame(columns=['Métodos', 'Média', 'STD', 'Limite Inferior', 'Limite Superior'])
dfResultClassifier['Métodos'] = ['Bagging', 'AdaBoost', 'RandomForest', 'HeterogeneousPooling']

def evaluateClassifiers(dataBase):
    """
    ## Classificadores
    ### Bagging
    """
    # define the model
    bagging = BaggingClassifier()

    grid = {'estimator__n_estimators': [10, 25, 50, 100]}
    model = bagging
    print('Model Evaluate... BaggingClassifier')
    bagScores = gridTest.GridTestModel(dataBase, model, grid)

    # Resultados
    print('****** Results ******\n')
    print('DataBase: ', dataBase.DESCR[4:18])
    print('Model - ', str(model)[0:17], '\n')
    bag_mean, bag_std, bag_inf, bag_sup = getResults.getResults(bagScores)


    dfResultClassifier.iloc[0] = ['Baggind', bag_mean, bag_std, bag_inf, bag_sup]
    dfResultClassifier

    """### AdaBoost"""

    # define the Model
    adaBoost = AdaBoostClassifier()

    # Z-score, Train, Test, gridSearch, CrossValidation
    grid = {'estimator__n_estimators': [10, 25, 50, 100]}
    model = adaBoost
    print('Model Evaluate... AdaBoostClassifier')
    adaScores = gridTest.GridTestModel(dataBase, model, grid)

    # Resultados
    print('****** Results ******\n')
    print('DataBase: ', dataBase.DESCR[4:18])
    print('Model - ', str(model)[0:18], '\n')
    ada_mean, ada_std, ada_inf, ada_sup = getResults.getResults(adaScores)

    dfResultClassifier.iloc[1] = ['Adaboost', ada_mean, ada_std, ada_inf, ada_sup]
    dfResultClassifier

    """### RandomForest"""

    # define the Model
    randomForest = RandomForestClassifier()

    # Model Evaluate
    grid = {'estimator__n_estimators': [10, 25, 50, 100]}
    model = randomForest
    print('Model Evaluate... RandomForestClassifier')
    rfScores = gridTest.GridTestModel(dataBase, model, grid)

    # Results
    print('****** Results ******\n')
    print('DataBase: ', dataBase.DESCR[4:18])
    print('Model - ', str(model)[0:22], '\n')
    rf_mean, rf_std, rf_inf, rf_sup = getResults.getResults(rfScores)

    dfResultClassifier.iloc[2] = ['RandomForest', rf_mean, rf_std, rf_inf, rf_sup]
    dfResultClassifier

    """### HeterogeneousPooling"""

    # define the model
    HeterogeneousModel = HP.HeterogeneousClassifier()

    # Model Evaluate
    grid = {'estimator__n_samples': [1,3,5,7]}
    model = HeterogeneousModel
    print('Model Evaluate... Heterogeneous Classifier')
    hpScores = gridTest.GridTestModel(dataBase, model, grid)

    # Results
    print('****** Results ******\n')
    print('DataBase: ', dataBase.DESCR[4:18])
    print('Model - ', str(model)[0:23], '\n')
    hp_mean, hp_std, hp_inf, hp_sup = getResults.getResults(hpScores)

    """## Results Classifiers Accuracy"""

    dfResultClassifier.iloc[3] = ['Heterogeneous', hp_mean, hp_std, hp_inf, hp_sup]
    dfResultClassifier

    import plotly.graph_objects as go
    scores = [bagScores, adaScores, rfScores, hpScores]
    scoresNames = ['Bagging', 'AdaBoost', 'RandomForest', 'Heterogeneous Polling']
    fig = go.Figure()
    for i in range(len(scores)):
      fig.add_trace(go.Box(y=scores[i], name=scoresNames[i]))
    fig.update_layout(
        yaxis_title='Accuracy',
        xaxis_title='Models',
        title=dataBase.DESCR[4:18]+' Models Performance  - Accuracy',
    )
    fig.show()

    """## Paired t-test and Wilcoxon Test"""

    from scipy.stats import ttest_rel, wilcoxon
    scores = [bagScores, adaScores, rfScores, hpScores]
    scoresNames = ['Bagging', 'AdaBoost', 'RandomForest', 'Heterogeneous Polling']
    dfPairTest = pd.DataFrame(columns=[0,1,2,3])
    for i in range(len(scores)):
      for j in range(len(scores)):
        if j == i:
          dfPairTest.at[i, j] = scoresNames[i]
        
        if j > i:
          print('Paired T Test', scoresNames[i], scoresNames[j])
          s,p = ttest_rel(scores[i],scores[j])
          print("t: %0.2f p-value: %0.8f\n" % (s,p))
          dfPairTest.at[i, j] = p
        

        if j < i :
          print ('Wilcoxon Test', scoresNames[i], scoresNames[j])
          s,p = wilcoxon (scores[i],scores[j])
          print("w: %0.2f p-value: %0.8f\n" % (s,p))
          dfPairTest.at[i, j] = p
      
    dfPairTest.columns = ['T1','T2','T3','T4']
    dfPairTest.index = ['w1', 'w2', 'w3', 'w4']
    
