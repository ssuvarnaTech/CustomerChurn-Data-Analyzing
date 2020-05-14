from collections import defaultdict

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, scale
from matplotlib import pyplot as plt
import numpy as np
import csv

data = pd.read_excel("/Users/sreevanisuvarna/Downloads/Customer_Churn.xlsx")


# plt.scatter(data['INCOME'], data['LEAVE'])
def categorize(dataX):
    for col in dataX.columns:
        if (dataX[col].dtype == np.object):
            dataX[col] = pd.Categorical(dataX[col])
    categoricals = dataX.select_dtypes(['category']).columns
    dataX[categoricals] = dataX[categoricals].apply(lambda x: x.cat.codes)


def getStats(fileName):
    rowTotal = 0
    Income = []
    houseValue = []
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0
    sum6 = 0
    sum7 = 0
    col1Dict = {}
    col2Dict = {}
    col3Dict = {}
    col4Dict = {}
    col5Dict = {}
    col1 = 'COLLEGE'
    col2 = 'INCOME'
    col3 = 'OVERAGE'
    col4 = 'LEFTOVER'
    col5 = 'HOUSE'
    col6 = 'HANDSET_PRICE'
    col7 = 'OVER_15MINS_CALLS_PER_MONTH'
    col8 = 'AVERAGE_CALL_DURATION'
    col9 = 'REPORTED_SATISFACTION'
    col10 = 'REPORTED_USAGE_LEVEL'
    col11 = 'CONSIDERING_CHANGE_OF_PLAN'
    col12 = 'LEAVE'
    with open(fileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rowTotal+=1
            Income.append(int(row[col2]))
            houseValue.append(int(row[col5]))
            if (row[col1] in col1Dict):
                col1Dict[row[col1]] += 1
            if (row[col1] not in col1Dict):
                col1Dict[row[col1]] = 1
            sum1 += int(row[col2])
            sum2 += int(row[col3])
            sum3 += int(row[col4])
            sum4+= int(row[col5])
            sum5+= int(row[col6])
            sum6+= int(row[col7])
            sum7 += int(row[col8])
            if (row[col9] in col2Dict):
                col2Dict[row[col9]] += 1
            if (row[col9] not in col2Dict):
                col2Dict[row[col9]] = 1

            if (row[col10] in col3Dict):
                col3Dict[row[col10]] += 1
            if (row[col10] not in col3Dict):
                col3Dict[row[col3]] = 1

            if (row[col11] in col4Dict):
                col4Dict[row[col11]] += 1
            if (row[col11] not in col4Dict):
                col4Dict[row[col11]] = 1

            if (row[col12] in col5Dict):
                col5Dict[row[col12]] += 1
            if (row[col12] not in col5Dict):
                col5Dict[row[col12]] = 1
        averageIncome = sum1/rowTotal
        averageOverage = sum2/rowTotal
        averageLeftover = sum3 / rowTotal
        averageHouse = sum4 / rowTotal
        averageHandsetPrice = sum5/rowTotal
        averageOverage15Mins = sum6 / rowTotal
        averageAverageCallDuration = sum7 / rowTotal
        houseValue.sort()
        Income.sort()
        medHouse = houseValue[int(len(houseValue)/2)]
        medIncome = Income[int(len(Income)/2)]
        with open(fileName, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['College', col1Dict])
            writer.writerow(['INCOME', averageIncome])
            writer.writerow(['OVERAGE', averageOverage])
            writer.writerow(['LEFTOVER', averageLeftover])
            writer.writerow(['HOUSE', averageHouse])
            writer.writerow(['Handset Price', averageHandsetPrice])
            writer.writerow(['Overage15mins', averageOverage15Mins])
            writer.writerow(['Average', averageAverageCallDuration])
            writer.writerow(['satisfaction', col2Dict])
            writer.writerow(['usage level', col3Dict])
            writer.writerow(['considering', col4Dict])
            writer.writerow(['leave', col5Dict])
            writer.writerow(['Median for House Value:' , medHouse])
            writer.writerow(['Median for Income', medIncome])




def main():
    trainX = data.drop(columns=["LEAVE"])

    trainY = data["LEAVE"]
    categorize(trainX)
    # print(trainX)
    print("normalizing data")
    # normalize the data
    for col in trainX.columns:
        if col == 'INCOME' or col == 'OVERAGE' or col == 'LEFTOVER' or col == 'HOUSE' or col == 'HANDSET_PRICE' or col == 'OVER_15MINS_CALLS_PER_MONTH' or col == 'AVERAGE_CALL_DURATION':
            trainX[col] = (trainX[col] - trainX[col].mean()) / trainX[col].std()
    y = pd.DataFrame(trainY)
    # print("getting linkage")
    # row_clusters = linkage(pdist(trainX, metric='euclidean'), method='complete')
    # indices = [i for i in range(trainX.shape[0])]
    # trainX = pd.DataFrame(trainX, index = indices)
    # row_dendr = dendrogram(row_clusters,
    #                        labels=indices)
    # plt.tight_layout()
    # plt.ylabel('Euclidean distance')
    # plt.show()
    # plt.close()
    print("doing KMeans")
    model = KMeans(n_clusters=6)
    model.fit(trainX)


    # plotting model outputs
    print("printing to file")
    clusterIndices = dict()
    index = 0
    for c in model.labels_:
        if c in clusterIndices:
            clusterIndices[c].append(index)
        else:
            clusterIndices[c] = []
            clusterIndices[c].append(index)
        index += 1
    index = 0
    trainx_data = data.to_numpy()
    for cluster in clusterIndices:
        name = "demofile" + str(cluster) + ".csv"
        f = open(name, 'w')
        columnList = ['COLLEGE', 'INCOME', 'OVERAGE', 'LEFTOVER', 'HOUSE', 'HANDSET_PRICE',
                      'OVER_15MINS_CALLS_PER_MONTH', 'AVERAGE_CALL_DURATION', 'REPORTED_SATISFACTION',
                      'REPORTED_USAGE_LEVEL', 'CONSIDERING_CHANGE_OF_PLAN', 'LEAVE']
        csvWriter = csv.writer(f)
        csvWriter.writerow(columnList)
        points = clusterIndices[cluster]
        print(trainx_data)
        for i in points:
            csvWriter.writerow(trainx_data[i])
            index += 1
        f.close()

    plt.show()
    plt.savefig("hello.png")
    plt.clf()


if __name__ == '__main__':
    main()
    getStats('demofile0.csv')
    getStats('demofile1.csv')
    getStats('demofile2.csv')
    getStats('demofile3.csv')
    getStats('demofile4.csv')
    getStats('demofile5.csv')
