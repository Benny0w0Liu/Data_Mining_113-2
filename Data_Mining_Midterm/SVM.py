import numpy as np
import csv


def SVM(X, y):
    C=1.0
    tolerance=0.003
    number_of_iter = 1000
    n_samples, n_features = X.shape
    alpha = np.zeros(n_samples)
    bias = 0
    weight = np.zeros(n_features)  # Initialize w with correct dimensions
        
    K = np.dot(X, X.T)

    for _ in range(number_of_iter):
        for i in range(n_samples):
            error_i = np.sign(np.dot(X[i], weight) + bias) - y[i]
            if (y[i] * error_i < -tolerance and alpha[i] < C) or (y[i] * error_i > tolerance and alpha[i] > 0):
                j = np.random.randint(0, n_samples)
                while j == i:
                    j = np.random.randint(0, n_samples)
                error_j = np.sign(np.dot(X[j], weight) + bias) - y[j]
                    
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                
                if y[i] != y[j]:
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
                
                alpha[j] -= y[j] * (error_i - error_j) / eta
                alpha[j] = np.clip(alpha[j], L, H)
                    
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue
                    
                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])
                    
                b1 = bias - error_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                b2 = bias - error_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                
                bias = (b1 + b2) / 2
        
    weight = np.sum((alpha * y)[:, None] * X, axis=0)
    return weight, bias
def predict(X,weight,bias):
    return np.sign(np.dot(X, weight) + bias)
def load_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    data = np.array(data[1:], dtype=np.float64)
    X, y = data[:, :-1], data[:, -1]
    return X, y
# Load training and test data
train_file_path = "./實驗B/train_data.csv"
test_file_path = "./實驗B/test_data.csv"
X_train, y_train = load_csv(train_file_path)
X_test, y_test = load_csv(test_file_path)

# Feature scaling
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Convert labels to -1 and 1
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Train SVM model
weight, bias=SVM(X_train, y_train)

# Predict on test set
y_pred = predict(X_test,weight,bias)

# Compute accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")