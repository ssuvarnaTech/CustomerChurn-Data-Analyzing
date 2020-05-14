#Importing the necessary libraries
import graphviz
import pandas as pd
from matplotlib import pyplot
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from subprocess import call
from graphviz import Source
from sklearn.tree import export_graphviz # will create a dot file
import xlrd
import pydot
import numpy as np
#read data
data = pd.read_excel("/Users/sreevanisuvarna/Downloads/Customer_Churn.xlsx")

def categorize(dataX):
  for col in dataX.columns:
      if(dataX[col].dtype == np.object):
          dataX[col] = pd.Categorical(dataX[col])
  categoricals = dataX.select_dtypes(['category']).columns
  dataX[categoricals] = dataX[categoricals].apply(lambda x : x.cat.codes)



def main():
    trainX = data.drop(columns=["LEAVE"])
    trainY= data["LEAVE"].to_numpy()
    categorize(trainX)
    print(trainX)
    print(trainY)
    trainYTemp = []
    for i in range(len(trainY)):
        if trainY[i] == 'STAY':
            trainYTemp.append(0)
        else:
            trainYTemp.append(1)
    trainY = np.array(trainYTemp)
    #normalize the data
    for col in trainX.columns:
        if col == 'INCOME' or col == 'OVERAGE' or col == 'LEFTOVER' or col == 'HOUSE' or col == 'HANDSET_PRICE' or col == 'OVER_15MINS_CALLS_PER_MONTH' or col == 'AVERAGE_CALL_DURATION':
           trainX[col] = (trainX[col] - trainX[col].mean())/trainX[col].std()
    #split the data
    X_train,X_test,y_train, y_test = train_test_split(trainX, trainY, test_size = 0.2, random_state = 1)
    #create the tree
    model = tree.DecisionTreeClassifier(max_depth = 10, max_features= 3)
    model.fit(X_train, y_train)
    #predict output
    y_predict = model.predict(X_test)
    print(accuracy_score(y_test, y_predict))

    #look at the confusion matrix
    matrix = pd.DataFrame(confusion_matrix(y_test, y_predict), columns = ['Predicted Leave', 'Predicted Stay'],
                 index = ['True Leave', 'True Stay'])
    print(matrix)
    #roc curve
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
    #export the tree
    tree_dot = StringIO()
    tree.export_graphviz(model, out_file= tree_dot, feature_names=trainX.columns, class_names = model.classes_)
    graph = pydot.graph_from_dot_data(tree_dot.getvalue())   # visualize the data
    graph[0].write_pdf("hw.pdf")
if __name__ == '__main__':
    main()






