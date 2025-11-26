import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():

    # --- Load & prep ---
    df = pd.read_csv("classical_residue_features.csv")

    # Features/labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # --- Model ---
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # --- Stratified 10-fold CV ---
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Per-fold metrics
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro"
    }

    cv_results = cross_validate(
        clf, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    # Print per-fold + aggregate
    print("Per-fold metrics:")
    for i in range(10):
        print(
            f"Fold {i+1}: "
            f"acc={cv_results['test_accuracy'][i]:.4f}, "
            f"prec_macro={cv_results['test_precision_macro'][i]:.4f}, "
            f"recall_macro={cv_results['test_recall_macro'][i]:.4f}, "
            f"f1_macro={cv_results['test_f1_macro'][i]:.4f}"
        )

    print("\nMean ± SD across folds:")
    for k, v in scoring.items():
        scores = cv_results[f"test_{v}"]
        print(f"{k}: {scores.mean():.4f} ± {scores.std():.4f}")

    # --- Pooled predictions for the full classification report ---
    y_oof = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    acc_oof = accuracy_score(y, y_oof)
    report_dict = classification_report(y, y_oof, output_dict=True)

    # Round metrics to 4 decimals
    rounded_report = {
        label: (
            {metric: round(value, 4) for metric, value in scores.items()}
            if isinstance(scores, dict)
            else round(scores, 4)
        )
        for label, scores in report_dict.items()
    }

    print(f"\nOOF Accuracy: {acc_oof:.4f}\n")
    print("OOF Classification Report:")
    report_df = pd.DataFrame(rounded_report).T
    print(report_df)


if __name__ == "__main__":
    main()
