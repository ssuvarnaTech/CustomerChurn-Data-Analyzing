import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas import Series, DataFrame
import scipy
from scipy.stats import spearmanr
from pylab import rcParams
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# read data
data = pd.read_excel("/Users/sreevanisuvarna/Downloads/Customer_Churn.xlsx")


def categorize(dataX):
    for col in dataX.columns:
        if (dataX[col].dtype == np.object):
            dataX[col] = pd.Categorical(dataX[col])
    categoricals = dataX.select_dtypes(['category']).columns
    dataX[categoricals] = dataX[categoricals].apply(lambda x: x.cat.codes)


def main():
    # STEP 1: DATA reading and understanding
    print(data.head())
    # STEP 2: NORMALIZE DATA
    trainX = data.drop(columns=["LEAVE"])
    trainY = data["LEAVE"].to_numpy()
    categorize(trainX)
    trainYTemp = []
    for i in range(len(trainY)):
        if trainY[i] == 'STAY':
            trainYTemp.append(0)
        else:
            trainYTemp.append(1)
    trainY = np.array(trainYTemp)
    # normalize the data
    for col in trainX.columns:
        if col == 'INCOME' or col == 'OVERAGE' or col == 'LEFTOVER' or col == 'HOUSE' or col == 'HANDSET_PRICE' or col == 'OVER_15MINS_CALLS_PER_MONTH' or col == 'AVERAGE_CALL_DURATION':
            trainX[col] = (trainX[col] - trainX[col].mean()) / trainX[col].std()
    #STEP 3 : Prepare the data(define indep/dependent variables)
    Y = trainY
    X = trainX
    print(X.head())
    #STEP 4: Split data
    X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.4, random_state= 20)
    #STEP 4: Define the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    #STEP 5: Testing the model
    prediction_test = model.predict(X_test)
    #STEP 6: Verify the accuracy
    print("Accuracy = ", metrics.accuracy_score(y_test,prediction_test))
    #STEP 7: Weights
    weights = pd.Series(model.coef_[0], index = X.columns.values)
    print(weights)
    #prints correlation
    # roc curve
    ns_probs = [0 for _ in range(len(y_test))]
    # predict probabilities
    lr_probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    pyplot.show()

if __name__ == '__main__':
    main()
