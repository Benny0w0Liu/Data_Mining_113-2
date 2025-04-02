import csv
import numpy as np

def PLA(training_datas):
    number_of_iter=10000
    data_vec=[]
    tag=[]
    for data in training_datas:
        vec=np.array(list(data.values())).astype(float);
        vec[-1]=1
        data_vec.append(vec)
        if(list(data.values())[-1]=='0'): tag.append(-1)
        else: tag.append(1)
    weight=np.zeros(len(data_vec[0]))
    for _ in range(0,number_of_iter):
        for i in range(0, len(data_vec)):
            if np.dot(data_vec[i],weight)*tag[i]<=0:
                weight += tag[i]*data_vec[i]
                break
    return weight
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
    weight=func(training_datas)
    for data in test_datas:
        vec=np.array(list(data.values())).astype(float)
        vec[-1]=1
        result=int(np.dot(vec,weight)>0)
        correct_answer+=int(int(data['Outcome'])==result)
    print("Experiment",experiment,'Accuracy:',correct_answer*100/len(test_datas),'%')
experiment("A",PLA)
experiment("B",PLA)