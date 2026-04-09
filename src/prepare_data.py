import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def prepare_dialect_data(file_path, train_ratio=0.75):
    df = pd.read_parquet(file_path)
    text_col, label_col = 'sentence', 'label'

    min_n = df[label_col].value_counts().min()
    df = df.groupby(label_col).sample(n=min_n, random_state=42)
    df_train, df_test = train_test_split(df, train_size=train_ratio, stratify=df[label_col], random_state=42)

    df_test.to_csv("test_dataset.csv", index=False)
    print(f"Saved {len(df_test)} sentences for clean validation to test_dataset.csv")

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3), binary=True, token_pattern=None)
    X_train_all = vectorizer.fit_transform(df_train[text_col])
    feature_names = vectorizer.get_feature_names_out()

    dialects = df_train[label_col].unique()
    n_dialects = len(dialects)

    presence_matrix = []
    for d in dialects:
        d_idx = np.where(df_train[label_col] == d)[0]
        presence_in_d = (X_train_all[d_idx].sum(axis=0) > 0).A1
        presence_matrix.append(presence_in_d)

    presence_matrix = np.array(presence_matrix)
    dialect_count_per_ngram = presence_matrix.sum(axis=0)

    for i, d in enumerate(dialects):
        mask = (presence_matrix[i] == True) & (dialect_count_per_ngram < n_dialects)
        selected_features = feature_names[mask]

        folder_path = f"dialect_{d}"
        os.makedirs(folder_path, exist_ok=True)

        with open(f"{folder_path}/unique_ngrams.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(selected_features))

        X_selected = X_train_all[:, mask]
        pd.DataFrame(X_selected.toarray(), columns=selected_features).to_csv(f"{folder_path}/binary_matrix.csv", index=False)
        np.savetxt(f"{folder_path}/labels.txt", np.where(df_train[label_col] == d, 1, -1), fmt='%d')

        print(f"Dialect {d}: {len(selected_features)} features saved (Train size: {len(X_selected.toarray())})")
