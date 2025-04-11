import numpy as np
import csv

def SVM(data_vec,tag, number_of_iter, C, tolerance):
    n_samples, n_features = data_vec.shape
    alpha = np.zeros(n_samples)
    bias = 0
    weight = np.zeros(n_features)
    #kernel
    K = np.dot(data_vec, data_vec.T)
    
    for _ in range(number_of_iter):
        for i in range(n_samples):
            #SMO Algorithm
            error_i = np.sign(np.dot(data_vec[i], weight) + bias) - tag[i]
            if (tag[i] * error_i < -tolerance and alpha[i] < C) or (tag[i] * error_i > tolerance and alpha[i] > 0):
                j = np.random.randint(0, n_samples)
                while j == i:
                    j = np.random.randint(0, n_samples)
                error_j = np.sign(np.dot(data_vec[j], weight) + bias) - tag[j]
                    
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                
                if tag[i] != tag[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                
                if L == H:
                    continue
                
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                
                alpha[j] -= tag[j] * (error_i - error_j) / eta
                alpha[j] = np.clip(alpha[j], L, H)
                    
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue
                    
                alpha[i] += tag[i] * tag[j] * (alpha_j_old - alpha[j])
                    
                b1 = bias - error_i - tag[i] * (alpha[i] - alpha_i_old) * K[i, i] - tag[j] * (alpha[j] - alpha_j_old) * K[i, j]
                b2 = bias - error_j - tag[i] * (alpha[i] - alpha_i_old) * K[i, j] - tag[j] * (alpha[j] - alpha_j_old) * K[j, j]
                
                bias = (b1 + b2) / 2

    weight = np.sum((alpha * tag)[:, None] * data_vec, axis=0)
    return weight, bias
def experiment(experiment, number_of_iter, C, tolerance):
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
    # Feature scaling
    mean = np.mean(training_datas, axis=0)
    std = np.std(training_datas, axis=0)
    training_datas = (training_datas - mean) / std
    test_datas = (test_datas - mean) / std
    # Convert labels to -1 and 1
    training_tags = np.where(training_tags == 0, -1, 1)
    test_tags = np.where(test_tags == 0, -1, 1)
    #get Accuracy
    weight,bias=SVM(training_datas, training_tags, number_of_iter, C, tolerance)
    test_pred = np.sign(np.dot(test_datas, weight) + bias)
    accuracy = np.mean(test_tags == test_pred)
    return accuracy*100