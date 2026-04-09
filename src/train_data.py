import pandas as pd
import numpy as np
import os

def train_manual_svm(S_matrix, y_vector, C=100, eta=0.01):
    n, m = S_matrix.shape
    w = np.zeros(m)
    b = 0.0
    lambd = 1 / C
    
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    for i in indices:
        s_i = S_matrix[i]
        y_i = y_vector[i]
        prediction = np.dot(s_i, w) + b
        
        if y_i * prediction < 1:
            w = w - eta * (lambd * w - y_i * s_i)
            b = b + eta * y_i
        else:
            w = w - eta * lambd * w
            
    return w, b

def run_training_for_all_dialects(C_val=100, eta_val=0.01):
    all_models_data = {}
    folders = [f for f in os.listdir('.') if f.startswith('dialect_') and os.path.isdir(f)]

    if not folders:
        return None

    for folder in folders:
        dialect_name = folder.split('_')[1]
        S = pd.read_csv(f"{folder}/binary_matrix.csv").to_numpy()
        y = np.loadtxt(f"{folder}/labels.txt")
        
        w, b = train_manual_svm(S, y, C=C_val, eta=eta_val)
        
        all_models_data[dialect_name] = {
            "w": w,
            "b": b,
            "features": pd.read_csv(f"{folder}/binary_matrix.csv", nrows=0).columns.tolist()
        }
    return all_models_data