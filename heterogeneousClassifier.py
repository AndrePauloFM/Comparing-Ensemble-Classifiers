
import numpy as np
from sklearn.utils import resample
from collections import Counter
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

"""# Class - Heterogeneous Polling"""

def mostCommon(estimatorsPredict):
  
    return [Counter(col).most_common() for col in zip(*estimatorsPredict)]
  
"""# Voting Classifiers"""
def votingClass(predMatrix, y_train):

    saida = np.array([])
    
    valuesFrequency = mostCommon(predMatrix)
    for j in range(predMatrix.shape[1]):
      if len(valuesFrequency[j])>1 and valuesFrequency[j][0][1] == valuesFrequency[j][1][1]:
        listaElementsTie = np.array([])
        for k in range(len(valuesFrequency[j])-1):
          if valuesFrequency[j][k][1] == valuesFrequency[j][k+1][1]:
            if k == 0 :
              listaElementsTie = np.append(listaElementsTie, valuesFrequency[j][k][0])
              listaElementsTie = np.append(listaElementsTie, valuesFrequency[j][k+1][0])
            else:
              listaElementsTie = np.append(listaElementsTie, valuesFrequency[j][k+1][0])
        # Após empate, retorna a classe mais votada mais frequente na base de treino
        saida = np.append(saida, compareFrequencyValues(y_train, listaElementsTie))
      else:
        # Retorna a classe mais votada entre os classificadores
        saida = np.append(saida, valuesFrequency[j][0][0])
    return saida

"""# Compare tie results"""
def compareFrequencyValues(y_train, elementsTie):
      indexElements = []
      orderArray = getMostfrequentValues(y_train)
      listValues = list(orderArray)
      for i in range(len(elementsTie)):
        indexElements.append(listValues.index(elementsTie[i]))
      mostFreq = min(indexElements)
      return orderArray[mostFreq]

def getMostfrequentValues(a):

    from collections import Counter
    mostfrequentValues = np.array([])
    b = Counter(a)
    arrayValoresFreq = b.most_common()
    for i in range(len(b)):
      mostfrequentValues = np.append(mostfrequentValues,arrayValoresFreq[i][0])
    return mostfrequentValues   

class HeterogeneousClassifier(BaseEstimator):

  estimatorBase = list()
  estimatorBase.append(KNeighborsClassifier(n_neighbors=1))
  estimatorBase.append(DecisionTreeClassifier())
  estimatorBase.append(GaussianNB())

  def __init__(self, base_estimator = estimatorBase,n_samples=None):
    
    self.base_estimator = base_estimator
    self.n_samples = n_samples
    self.ord = []
    self.estimators = []
    self.KNNclassifier = KNeighborsClassifier(n_neighbors=1)
    self.DTclassifier = DecisionTreeClassifier()
    self.NBclassifier = GaussianNB()
 

  def fit(self, X, y):

    self.ord = y
    for i in range(self.n_samples):
      if self.n_samples == 1:
        self.estimators.append(self.KNNclassifier.fit(X, y))
        self.estimators.append(self.DTclassifier.fit(X, y))
        self.estimators.append(self.NBclassifier.fit(X, y))
      else:
        X, y = resample(X,y, replace=True)
        self.estimators.append(self.KNNclassifier.fit(X, y))
        self.estimators.append(self.DTclassifier.fit(X, y))
        self.estimators.append(self.NBclassifier.fit(X, y))
    
    return self.estimators


  def predict(self, X):

      y_pred = []
      for i in range(self.n_samples):
        y_knn = np.array([self.estimators[0+(3*i)].predict(X)])
        y_DTree = np.array([self.estimators[1+(3*i)].predict(X)])
        y_naive = np.array([self.estimators[2+(3*i)].predict(X)])
        if i == 0:
          y_pred = np.vstack((y_knn, y_DTree, y_naive))
        else:
          y_pred = np.vstack((y_pred, y_knn, y_DTree, y_naive))
      y_predVot = votingClass(y_pred,self.ord)
      
      return y_predVot
