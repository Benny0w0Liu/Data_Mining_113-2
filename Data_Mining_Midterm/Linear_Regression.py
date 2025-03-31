import csv
import numpy as np

def linear_regression(test_data,training_datas):
    data_vec=[]
    tag=[]
    test_data=np.array(list(test_data.values())).astype(float)
    test_data[-1]=1
    for data in training_datas:
        vec=np.array(list(data.values()))
        vec[-1]=1
        data_vec.append(np.array(vec).astype(float))
        if(list(data.values())[-1]=='0'): tag.append(-1)
        else: tag.append(1)
    data_vec=np.array(data_vec)
    tag=np.array(tag)
    Transpose=np.transpose(data_vec)
    weight=np.matmul(np.matmul(np.linalg.inv(np.matmul(Transpose, data_vec)),Transpose),tag)
    return int(np.dot(test_data,weight)>0)
def experiment(experiment,func):
    #read train data
    training_datas=[]
    with open("./實驗"+experiment+"/train_data.csv", mode ='r') as file:    
        csvFile = csv.DictReader(file)
        for lines in csvFile:
            training_datas.append(lines)
    #read test data
    test_datas=[]
    with open("./實驗"+experiment+"/test_data.csv", mode ='r') as file:    
        csvFile = csv.DictReader(file)
        for lines in csvFile:
                test_datas.append(lines)
    #get Accuracy
    correct_answer=0
    for data in test_datas:
        correct_answer+=int(int(data['Outcome'])==func(data,training_datas))
    print("Experiment",experiment,'Accuracy:',correct_answer*100/len(test_datas),'%')
experiment("A",linear_regression)
experiment("B",linear_regression)