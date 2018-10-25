import numpy as np
import pandas as pd
import operator
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
    def getK(self):
        return self.k
    def predict(self,data): # data without labels
       distance = []
       for i in range(len(data)):
           dist = 0
           for j in range(len(self.learningData)):
               dist = np.linalg.norm(data[i] - self.learningData[j])
               distance.append((self.labels[j],dist))


 #      distance.sort(key=operator.itemgetter(1))
       x = (int)(len(distance) /len(data))
       y = 0
       z = x
       neighbors = []

       for i in range(len(data)):
           print("NR " + str (i))
           temp = []
           while(y<x):
               temp.append(distance[y])
               y = y + 1;
           temp.sort(key=operator.itemgetter(1))
           for j in range(self.k):
               print (temp[j])
           x = x + z




k = kNN(5,"iris.data.learning")

testData = np.array(pd.read_csv("iris.data.test",header=None))
test = testData[:,0:4]
k.predict(test)
