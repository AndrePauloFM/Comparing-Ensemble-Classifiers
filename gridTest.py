
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

def GridTestModel(dataBase, model, grid):

  # Data Base
  X = dataBase.data
  y = dataBase.target

  # Z-score
  scalar = StandardScaler()

  # Pipeline
  pipeline = Pipeline([('transformer', scalar), ('estimator', model)])

  # configure the cross-validation procedure
  rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=36851234)

  # configure Grid Search
  gs = GridSearchCV(estimator=pipeline, param_grid = grid, 
                    scoring='accuracy', cv = 4, verbose=0, refit=True)

  # Results
  scores = cross_val_score(gs, X, y, scoring='accuracy', 
                          cv = rkf)

  print('Done')
  return scores