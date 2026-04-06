import numpy as np
import os

def get_3grams(text):
    return [text[i:i+3] for i in range(len(text) - 2)]

def text_to_binary_vector(text, feature_list):
    m = len(feature_list)
    x = np.zeros(m)
    sentence_ngrams = set(get_3grams(text))

    for j in range(m):
        if feature_list[j] in sentence_ngrams:
            x[j] = 1
    return x

def predict_dialect(test_sentence, trained_models):
    scores = {}

    for dialect, model in trained_models.items():
        w = model['w']
        b = model['b']
        features = model['features']

        x = text_to_binary_vector(test_sentence, features)

        current_score = np.dot(x, w) + b
        scores[dialect] = current_score

    best_dialect = max(scores, key=scores.get)
    return best_dialect, scores[best_dialect]

if __name__ == "__main__":
    print("\n--- Dialect Recognition System ---")
    user_input = input("Enter a sentence to identify dialect: ")

    try:
        result, score = predict_dialect(user_input, trained_models)
        print(f"\nFinal Result: This sentence belongs to [{result}] (Score: {score:.4f})")
    except NameError:
        print("Error: 'trained_models' not found. Run training first!")
