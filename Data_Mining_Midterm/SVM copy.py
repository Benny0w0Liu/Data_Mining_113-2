import numpy as np
import csv

class SVM:
    def __init__(self, kernel='linear', C=1.0, tol=1e-3, max_iter=1000):
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0
        self.w = np.array([])  # Initialize w as an empty array

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.w = np.zeros(n_features)  # Initialize w with correct dimensions
        
        # Compute Kernel
        K = np.dot(X, X.T)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                E_i = self.predict(X[i]) - y[i]
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    E_j = self.predict(X[j]) - y[j]
                    
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                    
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                    
                    self.b = (b1 + b2) / 2
        
        self.w = np.sum((self.alpha * y)[:, None] * X, axis=0)
        
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
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
svm = SVM(C=1.0, max_iter=1000)
svm.fit(X_train, y_train)

# Predict on test set
y_pred = svm.predict(X_test)

# Compute accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")