# -*- coding: utf-8 -*-
"""*Combinado de Classificadores*

*André Paulo Ferreira Machado*

Este trabalho consiste em realizar uma comparação experimental entre um conjunto pré-definido de técnicas de aprendizado 
para classificação automática, baseadas na ideia de combinados de classificadores, aplicadas a alguns 
problemas de classificação. As técnicas escolhidas são: Bagging, AdaBoost, RandomForest e HeterogeneousPooling.

As bases de dados utilizadas: digits, wine e breast cancer.
"""

"""
Arquivo Jupter GoogleColab Localizado em:
    https://colab.research.google.com/drive/1lcFN7jYthD_4zRs_Y0FdGNANtQvBFsaC

"""

"""Os resultados de cada classificador são apresentados numa tabela contendo a média das acurácias 
obtidas em cada fold do ciclo externo, o desvio padrão e o intervalo de confiança a 95% de significância 
dos resultados, e também através do boxplot dos resultados de cada classificador em cada fold.

Os dados utilizados no conjunto de treino em cada rodada de teste são padronizados (normalizados o com z-score). 
Os valores de padronização obtidos nos dados de treino são utilizados para padronizar os dados do respectivo conjunto de teste.
O procedimento experimental de treinamento, validação e teste é realizado através de 3 rodadas de ciclos aninhados 
de validação e teste, com o ciclo interno de validação contendo 4 folds e o externo de teste com 10 folds. 
A busca em grade (grid search) do ciclo interno considera os os valores de hiperparâmetros definidos para cada técnica de aprendizado.
"""

from sklearn import datasets

import inquirer


"""# Avaliação dos Classificadores"""
import evaluateClassifiers as evaluate


"""# Seleção da Bases de Dados"""
questions = [
  inquirer.List('base',
                message="Selecione a Base de Dados",
                choices=['Digits', 'Wine', 'Breast Cancer'],
            ),
]
answers = inquirer.prompt(questions)

def functionDigits():
    print("Base de Dados Digits selecionada")
    dataBase = datasets.load_digits()
    return dataBase

def functionWine():
    print("Base de Dados Wine selecionada")
    dataBase = datasets.load_wine()
    return dataBase

def functionBreast():
    print("Base de Dados Breast Cancer selecionada")
    dataBase = datasets.load_breast_cancer()
    return dataBase

def default():
    print("Selecione Uma Base de Dados")

if __name__ == "__main__":
    switch = {
        "Digits": functionDigits,
        "Wine": functionWine,
        "Breast Cancer": functionBreast
    }

    case = switch.get(answers["base"], default)
    dataBase = case()
    evaluate.evaluateClassifiers(dataBase)
    
