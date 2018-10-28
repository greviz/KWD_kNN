import numpy as np
import pandas as pd
from kNN import kNN as kNN

k = kNN(3,"iris.data.learning")

testData = np.array(pd.read_csv("iris.data.test",header=None))
test = testData[:,0:4]
labels = testData [:,4]

print(k.score(test,labels))