import csv
import numpy as np

def PLA(data_vec,tag):
    number_of_iter=10000
    weight=np.zeros(len(data_vec[0]))
    bias=0
    for _ in range(0,number_of_iter):
        for i in range(0, len(data_vec)):
            if np.dot(data_vec[i],weight)*tag[i]+bias<=0:
                bias += tag[i]
                weight += tag[i]*data_vec[i]
                break
    return weight, bias
def experiment(experiment,func):
    #read train data
    with open("./實驗"+experiment+"/train_data.csv", 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    data = np.array(data[1:], dtype=np.float64)
    training_datas, training_tags = data[:, :-1], data[:, -1]
    #read test data
    with open("./實驗"+experiment+"/test_data.csv", mode ='r') as file:    
        reader = csv.reader(file)
        data = list(reader)
    data = np.array(data[1:], dtype=np.float64)
    test_datas, test_tags = data[:, :-1], data[:, -1]
    # Convert labels to -1 and 1
    training_tags = np.where(training_tags == 0, -1, 1)
    test_tags = np.where(test_tags == 0, -1, 1)
    #get Accuracy
    weight,bias=func(training_datas, training_tags)
    test_pred = np.sign(np.dot(test_datas, weight) + bias)
    accuracy = np.mean(test_tags == test_pred)
    print("Experiment",experiment,'Accuracy:',accuracy*100,'%')
experiment("A",PLA)
experiment("B",PLA)