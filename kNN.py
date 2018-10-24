import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist

class kNN:
    def __init__(self,k,nameOfLearningDataFile):
        self.k=k
        self.fullLearningData=np.array(pd.read_csv(nameOfLearningDataFile,header=None))
        self.learningData=self.fullLearningData[:,0:4]
        self.labels=self.fullLearningData[:,4]
    def getLearningData(self):
        return self.learningData
    def getLabels(self):
        return self.labels


k = kNN(5,"iris.data.learning")
x = k.getLabels()
print(x)
y=k.getLearningData()
print(y)
