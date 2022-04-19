import numpy as np
from scipy import stats

def getResults(scores):
  print (scores)

  mean = scores.mean()
  std = scores.std()
  inf, sup = stats.norm.interval(0.95, loc=mean, 
                                scale=std/np.sqrt(len(scores)))

  print("\nMean Accuracy: %0.3f Standard Deviation: %0.3f" % (mean, std))
  print ("Accuracy Confidence Interval (95%%): (%0.2f, %0.2f)\n" % 
        (inf, sup)) 

  return mean, std, inf, sup