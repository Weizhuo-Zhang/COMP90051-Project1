import sys
import pandas as pd
import numpy as np
print(sys.argv)

groundTruePath = sys.argv[1]
predictionPath = sys.argv[2]


groundTrue = pd.read_csv(groundTruePath, sep='\t', header=None)
predicted = pd.read_csv(predictionPath)


groundTrueList = list(groundTrue[0])
predictedList = list(predicted['Predicted'])

correct = 0
for i in range(len(groundTrueList)):
    if groundTrueList[i] == predictedList[i]:
        correct += 1

print('accuracy: {:.2%}'.format(correct/len(groundTrueList)))
