import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime

class RidgeRegression:
    def __init__(self):
        return
    
    def fit(self, X_train, Y_train, LAMBDA):
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]
        W = np.linalg.inv(X_train.transpose()@X_train + LAMBDA * np.identity(X_train.shape[1])) \
                     @(X_train.T)@Y_train
        return W

    def fitGrad(self, X_train, Y_train, LAMBDA, learningRate, maxEpochs=100, batchSize=128):
        W = np.random.rand(X_train.shape[1])
        lastLoss = 1e10
        for epoch in range(maxEpochs):
            arr = np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            X_train, Y_train = X_train[arr], Y_train[arr]
            numMinibatch = int(np.ceil(X_train.shape[0]/batchSize))
            for i in range(numMinibatch):
                index = i*batchSize
                X_train_sub = X_train[index:index+batchSize]
                Y_train_sub = Y_train[index:index+batchSize]
                grad = X_train_sub.T@(X_train_sub@W-Y_train_sub) + LAMBDA*W
                W = W - learningRate*grad
            newLoss = self.computeRSS(self.predict(W, X_train), Y_train)
            if np.abs(newLoss-lastLoss) < 1e-10:
                break
            lastLoss = newLoss
        return W

    def predict(self, W, X_test):
        return X_test@W

    def computeRSS(self, Y_predicted, Y_test):
        return np.sum((Y_predicted-Y_test)**2)/Y_predicted.shape[0] 

    def getTheBestLambda(self, X, Y):
        def crossValid(numFolds, LAMBDA):
            kf = KFold(n_splits=numFolds)
            sumRSS = 0
            for trainID, valID in kf.split(X):
                W = self.fit(X[trainID], Y[trainID], LAMBDA = LAMBDA)
                Y_pred = self.predict(W, X[valID])
                sumRSS += self.computeRSS(Y_pred, Y[valID])
            aveRSS = sumRSS/numFolds
            return aveRSS

        def rangeScan(bestLAMBDA, minRSS, LAMBDAVals):
            for curLAMBDA in LAMBDAVals:
                aveRSS = crossValid(numFolds=5, LAMBDA=curLAMBDA)
                if aveRSS < minRSS:
                    bestLAMBDA = curLAMBDA
                    minRSS = aveRSS
            return bestLAMBDA, minRSS

        bestLAMBDA, minRSS = rangeScan(bestLAMBDA=1, minRSS=1e10, LAMBDAVals=range(50))
        LAMBDAVals = [k/1000 for k in range(max(0,(bestLAMBDA-1)*1000), (bestLAMBDA+1)*1000, 1)]
        bestLAMBDA, minRSS = rangeScan(bestLAMBDA=bestLAMBDA, minRSS=minRSS, LAMBDAVals=LAMBDAVals)
        return bestLAMBDA

def normalizeAndAddOnes(X):
    X = np.array(X)
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    X_normalized = (X-X_min)/(X_max-X_min)
    ones = np.ones(X_normalized.shape[0])
    return np.column_stack((ones,X_normalized))

if __name__=="__main__":
    data = pd.read_csv("datasets/x28.csv", delimiter='\s+')
    data = data.iloc[:, 1:].values
    data = normalizeAndAddOnes(data)
    X_train, X_test = data[:50, :-1], data[50:, :-1]
    Y_train, Y_test = data[:50, -1], data[50:, -1]

    LR = RidgeRegression()
    LAMBDA = LR.getTheBestLambda(X_train, Y_train)
    LAMBDA = LR.getTheBestLambda(X_train, Y_train)
    # W = LR.fit(X_train, Y_train, LAMBDA)
    W = LR.fitGrad(X_train, Y_train, LAMBDA, 0.001, maxEpochs = 5000)
    Y_pred = LR.predict(W, X_test)
    with open("Session 1/Ridge Regression/result.txt", "w") as f:
        f.write(f'{datetime.now()}\n')
        f.write(f'Best LAMBDA: {LAMBDA}\n')
        f.write(f'RSS: {LR.computeRSS(Y_pred, Y_test)}')