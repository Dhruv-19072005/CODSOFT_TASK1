"""
CODSOFT - TASK 1: MOVIE GENRE CLASSIFICATION
==============================================
Dataset: Genre Classification Dataset IMDb (Kaggle)
https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

Approach:
 1. Load train_data.txt (format: ID ::: TITLE ::: GENRE ::: DESCRIPTION)
 2. Clean the plot summary text
 3. Convert text -> numeric features using TF-IDF
 4. Train 3 classifiers: Naive Bayes, Logistic Regression, Linear SVM
 5. Compare accuracy and pick the best model
 6. Predict genres for test_data.txt and save results
 7. Save trained model + vectorizer for reuse
"""

import re
import string
import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# CONFIG - update these paths to where you placed the dataset files
# -------------------------------------------------------------------
TRAIN_PATH = "train_data.txt"
TEST_PATH = "test_data.txt"                 # optional (no genre labels)
TEST_SOLUTION_PATH = "test_data_solution.txt"  # optional (has genre labels, for extra evaluation)

RANDOM_STATE = 42


# -------------------------------------------------------------------
# STEP 1: LOAD DATA
# -------------------------------------------------------------------
def load_data(path, has_genre=True):
    """
    The CodSoft/Kaggle dataset uses ' ::: ' as a separator.
    Train format : ID ::: TITLE ::: GENRE ::: DESCRIPTION
    Test format  : ID ::: TITLE ::: DESCRIPTION
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ::: ")
            if has_genre and len(parts) == 4:
                rows.append(parts)
            elif not has_genre and len(parts) == 3:
                rows.append(parts)

    if has_genre:
        df = pd.DataFrame(rows, columns=["ID", "TITLE", "GENRE", "DESCRIPTION"])
    else:
        df = pd.DataFrame(rows, columns=["ID", "TITLE", "DESCRIPTION"])
    return df


# -------------------------------------------------------------------
# STEP 2: CLEAN TEXT
# -------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep only letters
    text = re.sub(r"\s+", " ", text).strip()               # remove extra spaces
    return text


def main():
    print("Loading training data...")
    train_df = load_data(TRAIN_PATH, has_genre=True)
    print(f"Loaded {len(train_df)} rows")
    print(train_df["GENRE"].value_counts())

    print("\nCleaning text...")
    train_df["clean_plot"] = train_df["DESCRIPTION"].apply(clean_text)

    # -----------------------------------------------------------
    # STEP 3: TRAIN / VALIDATION SPLIT
    # -----------------------------------------------------------
    X = train_df["clean_plot"]
    y = train_df["GENRE"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # -----------------------------------------------------------
    # STEP 4: TF-IDF VECTORIZATION
    # -----------------------------------------------------------
    print("\nVectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # -----------------------------------------------------------
    # STEP 5: TRAIN MULTIPLE MODELS AND COMPARE
    # -----------------------------------------------------------
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=10),
        "Linear SVM": LinearSVC(max_iter=5000),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_val_tfidf)
        acc = accuracy_score(y_val, preds)
        results[name] = acc
        trained_models[name] = model
        print(f"{name} Validation Accuracy: {acc:.4f}")

    # -----------------------------------------------------------
    # STEP 6: PICK BEST MODEL
    # -----------------------------------------------------------
    best_name = max(results, key=results.get)
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name} (accuracy = {results[best_name]:.4f})")

    best_preds = best_model.predict(X_val_tfidf)
    print("\nClassification Report (best model):")
    print(classification_report(y_val, best_preds))

    # -----------------------------------------------------------
    # STEP 7: SAVE PLOTS (accuracy comparison + confusion matrix)
    # -----------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(results.keys(), results.values(), color=["#4C72B0", "#DD8452", "#55A868"])
    plt.ylabel("Validation Accuracy")
    plt.title("Model Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close()

    top_genres = y_val.value_counts().nlargest(10).index
    mask = y_val.isin(top_genres)
    cm = confusion_matrix(y_val[mask], pd.Series(best_preds, index=y_val.index)[mask], labels=top_genres)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=top_genres, yticklabels=top_genres, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name} (Top 10 genres)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------
    # STEP 8: SAVE MODEL + VECTORIZER FOR REUSE
    # -----------------------------------------------------------
    with open("genre_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    print("\nSaved genre_model.pkl and tfidf_vectorizer.pkl")

    # -----------------------------------------------------------
    # STEP 9: PREDICT ON test_data.txt (if provided)
    # -----------------------------------------------------------
    try:
        test_df = load_data(TEST_PATH, has_genre=False)
        test_df["clean_plot"] = test_df["DESCRIPTION"].apply(clean_text)
        test_tfidf = tfidf.transform(test_df["clean_plot"])
        test_df["PREDICTED_GENRE"] = best_model.predict(test_tfidf)
        test_df[["ID", "TITLE", "PREDICTED_GENRE"]].to_csv("test_predictions.csv", index=False)
        print("Saved predictions to test_predictions.csv")
    except FileNotFoundError:
        print(f"\n({TEST_PATH} not found - skipping test prediction step)")

    return best_model, tfidf, results


def predict_genre(plot_summary, model, vectorizer):
    """Utility: predict genre for a single custom plot summary."""
    cleaned = clean_text(plot_summary)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]


if __name__ == "__main__":
    best_model, tfidf, results = main()

    # Example usage on a custom plot summary
    sample_plot = (
        "A young wizard discovers his magical heritage and attends a school "
        "of witchcraft, where he must battle a dark lord who killed his parents."
    )
    predicted = predict_genre(sample_plot, best_model, tfidf)
    print(f"\nSample prediction -> Genre: {predicted}")