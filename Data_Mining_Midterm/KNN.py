import csv
def KNN(test_data,training_datas, K):
    #calculate distance(Euclidean distance)
    data_distances=[]
    for data in training_datas:
        distance=0
        for key in data:
            if(key!='Outcome'):
                distance+=pow(float(data[key])-float(test_data[key]),2)
        data_distances.append({'dis':pow(distance,0.5),'Outcome':data['Outcome']})
    #get K closest data
    data_distances = sorted(data_distances, key=lambda d: d['dis'])
    Outcome=[0,0]
    for data in data_distances[0:K]:
        Outcome[int(data['Outcome'])]+=1
    #choose the greater class
    return int(Outcome[1]>Outcome[0])

def experiment(experiment, K):
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
        correct_answer+=int(int(data['Outcome'])==KNN(data,training_datas,K))
    return correct_answer*100/len(test_datas)
print("Experiment A Accuracy(K=7):",KNN_experiment("A",7),'%')
print("Experiment B Accuracy(K=7):",KNN_experiment("B",7),'%')
