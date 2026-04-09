import os
import shutil
import argparse
import pandas as pd
import numpy as np
from src.prepare_data import prepare_dialect_data
from src.train_data import run_training_for_all_dialects
from src.inference import predict_dialect
from src.error_report import generate_error_summary

def clean_old_data():
    folders = [f for f in os.listdir('.') if f.startswith('dialect_') and os.path.isdir(f)]
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")
    for folder in folders:
        shutil.rmtree(folder)
    if os.path.exists("test_dataset.csv"):
        os.remove("test_dataset.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--c", type=float, default=100.0)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    dataset_url = "hf://datasets/statworx/swiss-dialects/data/train-00000-of-00001-70a98163b61f1b71.parquet"
    clean_old_data()
    prepare_dialect_data(dataset_url, train_ratio=args.ratio)
    trained_models = run_training_for_all_dialects(C_val=args.c, eta_val=args.eta)

    if trained_models is None:
        return

    if os.path.exists("test_dataset.csv"):
        generate_error_summary(trained_models)

    print("\n" + "="*50)
    print(f"SYSTEM READY (C={args.c}, Ratio={args.ratio}, Eta={args.eta})")
    print("="*50)

    if args.interactive:
        while True:
            sentence = input("\nEnter Swiss sentence (or 'exit'): ")
            if sentence.lower() == 'exit': 
                break
            label, score = predict_dialect(sentence, trained_models)
            print(f"Result: {label} (Score: {score:.2f})")

if __name__ == "__main__":
    main()
