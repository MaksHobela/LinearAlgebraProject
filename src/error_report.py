import pandas as pd
import numpy as np
import os
from src.inference import predict_dialect
from src.train_data import run_training_for_all_dialects

def load_trained_models():
    models = {}
    folders = [f for f in os.listdir('.') if f.startswith('dialect_') and os.path.isdir(f)]

    if not folders:
        print("Error: data folders not found")
        return None

    for folder in folders:
        d_name = folder.split('_')[1]
        try:
            return run_training_for_all_dialects(C_val=100)
        except Exception as e:
            print(f"Error loading {d_name}: {e}")
    return models

def generate_error_summary(trained_models=None):
    if not os.path.exists("test_dataset.csv"):
        print("Error: test_dataset.csv wasn't found!")
        return

    if trained_models is None:
        print("Authomatical launch...")
        trained_models = load_trained_models()
        if not trained_models: return

    df_test = pd.read_csv("test_dataset.csv")
    error_counts = {}
    correct = 0

    print(f"\nValidating {len(df_test)} sentences...")

    for _, row in df_test.iterrows():
        true_l = row['label']
        sentence = row['sentence']

        pred_l, _ = predict_dialect(sentence, trained_models)

        if true_l == pred_l:
            correct += 1
        else:
            pair = (true_l, pred_l)
            error_counts[pair] = error_counts.get(pair, 0) + 1

    print("\n" + "="*50)
    print(f"RESULT: {(correct/len(df_test))*100:.2f}% accuracy")
    print("="*50)
    print("MOST COMMON DIALECT ERRORS:")
    print("="*50)

    sorted_err = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    for (t, p), count in sorted_err:
        print(f"Is {t}, was identified as {p}: {count}")
    print("="*50)

if __name__ == "__main__":
    generate_error_summary()
